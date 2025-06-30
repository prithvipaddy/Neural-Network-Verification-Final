{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE BangPatterns #-}
-- {-# LANGUAGE LiquidHaskell #-}

module Main where

import Data.List (transpose)
import Data.Aeson (FromJSON, decode, encode)
import qualified Data.ByteString.Lazy as B (readFile, writeFile)
import GHC.Generics (Generic)
-- import Data.Vector (toList)
import Data.Matrix as M
import Data.Maybe (fromMaybe)
import System.Environment (getArgs)
import System.IO
import Control.Parallel.Strategies ( parMap, rpar, using, parListChunk, rdeepseq )
-- import Data.Sparse.SpMatrix as SP
-- import Numeric.Sparse.Matrix (smvm)
import qualified Data.Vector.Unboxed as U
-- import Data.Ix (range)

type Zonotope = Matrix Double

data LayerInfo = LayerInfo
  { layerName          :: String
  , layerType          :: String
  , inputShape         :: Maybe (Int,Int)
  , numFilters         :: Maybe Int
  , kernelSize         :: Maybe [Int]
  , activationFunction :: Maybe String
  , filters            :: Maybe [[[[Double]]]]
  , biases             :: Maybe [Double]
  , poolSize           :: Maybe [Int]
  , units              :: Maybe Int
  , weights            :: Maybe [[Double]]
  } deriving (Show, Generic)
instance FromJSON LayerInfo

data ImageData = ImageData
  { imageValues          :: [[[Double]]]
  , imageClass           :: Int
  , imageDimensions      :: (Int,Int,Int)
  } deriving (Show, Generic)
instance FromJSON ImageData

-- CONVOLUTION (from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8418593)   
generateBiasMatrixForConv :: [U.Vector Double] -> Double -> U.Vector Double
generateBiasMatrixForConv z b = let
  !size = U.length (head z)
  in U.replicate size b

-- Generate sparse convolution matrix using CSR format
-- sparse matrix, represented as a list of (rowIndex, colIndex, value) tuples
generateSparseConvMatrix :: [[Double]] -> (Int, Int) -> [(Int, Int, Double)]
generateSparseConvMatrix kernel (inpRows, inpCols) =
  let !kernelRows = length kernel
      !kernelCols = length (head kernel)
      !yRows = inpRows - kernelRows + 1
      !yCols = inpCols - kernelCols + 1
      !outputSize = yRows * yCols
      !kernelFlat = U.fromList (concat kernel)

      -- Parallel generation of matrix entries
      !entries = concat $ parMap rdeepseq (generateEntriesForOutputIndex (inpCols, kernelRows, kernelCols, kernelFlat))
                 [0..outputSize-1]
      in entries
  -- in fromListSM (outputSize, inputSize) entries

generateEntriesForOutputIndex :: (Int, Int, Int, U.Vector Double) -> Int -> [(Int, Int, Double)]
generateEntriesForOutputIndex (inpCols, kr, kc, kernelFlat) outIdx =
  let (!i, !j) = outIdx `divMod` (inpCols - kc + 1)
      !baseRow = i * inpCols + j
  in [ (outIdx, baseRow + dj + di * inpCols, kernelFlat `U.unsafeIndex` (di * kc + dj))
     | di <- [0..kr-1]
     , dj <- [0..kc-1]
     ]

-- | Multiply a sparse matrix, represented as a list of (rowIndex, colIndex, value) tuples,
--   with a dense vector.
smvm :: Int -> Int -> [(Int, Int, Double)] -> U.Vector Double -> U.Vector Double
smvm nRows nCols nonZeroElems vec
  | nCols /= U.length vec =
      error $ "Dimension mismatch: matrix has " ++ show nCols ++ " columns, but vector has " ++ show (U.length vec) ++ " elements."
  | otherwise = U.accum (+) (U.replicate nRows 0) contributions
  where
    contributions = [(i, val * (vec U.! j)) | (i, j, val) <- nonZeroElems]

test3Conv :: Bool
test3Conv = let
  k = [[1,1],[1,1]] :: [[Double]]
  vec = U.fromList [1..8] :: U.Vector Double
  expected = U.fromList [14,18,22]
  w = generateSparseConvMatrix k (2,4)
  result = smvm 3 8 w vec
  in result==expected

applySingleChannelConvolution :: [U.Vector Double] -> (Int,Int) -> [[Double]] -> [U.Vector Double]
applySingleChannelConvolution zonotope (inpImgRows,inpImgCols) kernel = let
    !kRows = length kernel
    !kCols = length (head kernel)
    !yRows = inpImgRows - kRows + 1
    !yCols = inpImgCols - kCols + 1
    !wF = generateSparseConvMatrix kernel (inpImgRows,inpImgCols)
    !newZonotope = map (smvm (yRows*yCols) (inpImgRows*inpImgCols) wF) zonotope
    in newZonotope

-- applySingleChannelConvolution :: Matrix Double -> (Int,Int) -> [[Double]] -> Matrix Double
-- applySingleChannelConvolution zonotope (inpImgRows,inpImgCols) kernel = let
--     !wF = M.fromLists (generateWeightMatrixForConv kernel (inpImgRows,inpImgCols))
--     !newZonotope = M.multStd wF zonotope
--     in newZonotope

-- scalar composotion of the entire depth of the zonotopes to form a single zonotope
-- one scalar composition for each filter 
-- eg. if initial depth = 3, newZ is sum of the 3 zonotopes, and if numFilters = 32, new depth = 32
-- Function to sum corresponding elements of each sublist of vectors
sumVectors :: [[U.Vector Double]] -> [U.Vector Double]
sumVectors c = map (foldr1 (U.zipWith (+))) (transpose' c)
  where
    -- Transpose the list of lists of vectors to group corresponding vectors together
    transpose' :: [[U.Vector Double]] -> [[U.Vector Double]]
    transpose' [] = repeat []  -- To handle the empty case
    transpose' (x:xs) = zipWith (:) x (transpose' xs)

applyConvolutionPerFilter' :: [[U.Vector Double]] -> (Int,Int) -> [[[Double]]] -> [U.Vector Double]
applyConvolutionPerFilter' zonotope (inpImgRows,inpImgCols) kernel = let
    !convolutions = parMap rpar (\ (z1, k1) -> applySingleChannelConvolution z1 (inpImgRows, inpImgCols) k1) (zip zonotope kernel)
    !composedConvolutions = sumVectors convolutions
    in composedConvolutions

applyConvolutionPerFilter :: [[U.Vector Double]] -> (Int,Int) -> [[[Double]]] -> Double -> [U.Vector Double]
applyConvolutionPerFilter zonotope (inpImgRows,inpImgCols) kernel bias = let
    !convolved = applyConvolutionPerFilter' zonotope (inpImgRows,inpImgCols) kernel
    !biasVector = generateBiasMatrixForConv convolved bias
    !newCenter = U.zipWith (+) biasVector (head convolved)
    !final = newCenter : tail convolved
    in final

-- length of kernel and bias should be same
applyConvolution :: [[U.Vector Double]] -> (Int,Int) -> [[[[Double]]]] -> [Double] -> [[U.Vector Double]]
applyConvolution zonotope (inpImgRows,inpImgCols) kernel bias = let
  !convolutions = parMap rpar (uncurry (applyConvolutionPerFilter zonotope (inpImgRows, inpImgCols))) (zip kernel bias)
  in convolutions

-- 2D Convolution for a flattened image without padding
conv2dFlattened :: Num a => [a] -> Int -> Int -> [a] -> Int -> Int -> [a]
conv2dFlattened image height width kernel kh kw =
    [ sum $ zipWith (*) (extractSubMatrixFlattened image width i j kh kw) kernel
    | i <- [0..(height - kh)], j <- [0..(width - kw)] ]

-- Extract a sub-matrix from the flattened image for convolution at position (i, j)
extractSubMatrixFlattened :: Num a => [a] -> Int -> Int -> Int -> Int -> Int -> [a]
extractSubMatrixFlattened image width i j kernelHeight kernelWidth =
    [ image !! ( (i + di) * width + (j + dj) )
    | di <- [0..(kernelHeight - 1)], dj <- [0..(kernelWidth - 1)] ]

x1Convolution :: [[U.Vector Double]]
x1Convolution = [[[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8]],[[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8]],[[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8]]]

k1Convolution :: [[[[Double]]]]
k1Convolution = [[[[1,1],[1,1]],[[1,1],[1,1]],[[1,1],[1,1]]]]

test1Convolution :: [[U.Vector Double]]
test1Convolution = let
  result = applyConvolution x1Convolution (2,4) k1Convolution [10]
  -- expected = [[U.fromList [14,18,22]]]
  in result

test2Convolution :: Bool
test2Convolution = let
  img = [1,2,3,4,5,6,7,8] :: [Double]
  k = [1,1,1,1] :: [Double]
  result = conv2dFlattened img 2 4 k 2 2
  expected = [14,18,22]
  in result == expected
{-
Example usage of generateWPooling
p = 2, q = 1
x:
[0,1,2,3]
[4,5,6,7]
[8,9,10,11]
[12,13,14,15]
xv = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

groups: [[0,4],[1,5],[2,6],[3,7],[8,12],[9,13],[10,14],[11,15]] 
NOTE FOR GROUPS: these will always be indices, it just so happens that x has been defined with values matching indices
groupMap = [0,4,1,5,2,6,3,7,8,12,9,13,10,14,11,15]

when i = 0, checks the value for groupMap at index 0.
since it is 0, it will input 1 into the first position of row 0.

i = 1, checks the value for groupMax at index 1.
passes through all j values upto 4, inserting '0' in those positions. at index 4, it will insert a '1'
this indicates that the 5th element from xv should be the 2nd element in xMP

Similarly done, upto i = 15, and wMP is constructed.
-}
-- AVERAGE POOLING
poolingGroups :: Int -> Int -> [[Double]] -> [[Int]]
poolingGroups p q x =
  let !rows = length x
      !cols = length (head x)
      -- groupIndices finds the indices for all elements in a pooling box.
      -- The input to the function is the top-left element of each box
      groupIndices i j = [ (i + di) * cols + (j + dj) | di <- [0 .. p - 1], dj <- [0 .. q - 1] ]
      -- Below, the groupIndices for the top-left element of every pooling group is found
      !groups = [ groupIndices (i * p) (j * q) | i <- [0 .. (rows `div` p) - 1], j <- [0 .. (cols `div` q) - 1] ]
   in groups

generateSparsePoolingMatrix :: Int -> Int -> [[Double]] -> [(Int, Int, Double)]
generateSparsePoolingMatrix p q x =
  let !rows = length x
      !cols = length (head x)
      !numOutputRows = rows `div` p
      !numOutputCols = cols `div` q
      !outputSize = numOutputRows * numOutputCols
      !poolSize' = fromIntegral (p * q)

      -- Parallel generation of entries
      !entries = concat $ parMap rdeepseq
                (\outIdx -> generatePoolingEntries p q cols numOutputCols outIdx poolSize')
                [0..outputSize-1]
  in entries

generatePoolingEntries :: Int -> Int -> Int -> Int -> Int -> Double -> [(Int, Int, Double)]
generatePoolingEntries p q cols numOutputCols outIdx poolSize' =
  let (!i, !j) = outIdx `divMod` numOutputCols
      !inputStartRow = i * p
      !inputStartCol = j * q
      !scaleFactor = 1.0 / poolSize'
  in [ (outIdx, (inputStartRow + di) * cols + (inputStartCol + dj), scaleFactor)
     | di <- [0..p-1]
     , dj <- [0..q-1]
     ]

averagePooling :: Int -> Int -> [[Double]] -> U.Vector Double
averagePooling p q x =
  let !rows = length x
      !cols = length (head x)
      !inputSize = rows * cols
      !outputSize = (rows `div` p) * (cols `div` q)
      -- Create sparse matrix and flatten input
      !sparseMat = generateSparsePoolingMatrix p q x
      !flatInput = U.fromList (concat x)
      -- Perform sparse matrix-vector multiplication
      !result = smvm outputSize inputSize sparseMat flatInput
  in result

applySinglePointAveragePooling :: Int -> Int -> U.Vector Double -> (Int,Int) -> U.Vector Double
applySinglePointAveragePooling p q point (rows,cols) = let
  !x = M.toLists (M.fromList rows cols (U.toList point))
  !pooled = averagePooling p q x
  in pooled

applyAllPointsAveragePooling :: Int -> Int -> [U.Vector Double] -> (Int,Int) -> [U.Vector Double]
applyAllPointsAveragePooling p q zonotope (imgRows,imgCols) = let
  !avgPooledPoints = map (\point -> applySinglePointAveragePooling p q point (imgRows,imgCols)) zonotope `using` parListChunk 100 rpar
  in avgPooledPoints

applyAveragePooling :: Int -> Int -> [[U.Vector Double]] -> (Int,Int) -> [[U.Vector Double]]
applyAveragePooling p q zonotope (imgRows,imgCols) = let
  !avgPooled = parMap rpar (\z -> applyAllPointsAveragePooling p q z (imgRows,imgCols)) zonotope
  in avgPooled

-- AVERAGE POOLING TESTS
x1AveragePooling :: [U.Vector Double]
x1AveragePooling = [U.fromList [1..8],U.fromList [1..8]]

test1AveragePooling :: Bool
test1AveragePooling = let
  pooled = applyAveragePooling 2 2 [x1AveragePooling] (2,4)
  expected = [[U.fromList [3.5,5.5],U.fromList [3.5,5.5]]]
  in pooled==expected

-- RELU (SHEARING)
reluUpper :: [Double] -> Double
reluUpper eq = let
  center = head eq
  generators = tail eq
  in center + sum (map abs generators)

reluLower :: [Double] -> Double
reluLower eq = let
  center = head eq
  generators = tail eq
  in center - sum (map abs generators)

composeLambdaAndNuRelu :: [Double] -> [Double]
composeLambdaAndNuRelu eq = let
    !u = reluUpper eq
    !l = reluLower eq
    !lambda = u / (u-l)
    !nu = (u * (1 - lambda)) / 2
    !lambdaScaled = map (* lambda) eq ++ [0]
    !lengthLambdaScaled = length lambdaScaled
    !nScaled = [nu] ++ replicate (lengthLambdaScaled - 2) 0 ++ [nu]
    in zipWith (+) lambdaScaled nScaled

applyReluPerDimension :: [Double] -> [Double]
applyReluPerDimension eq = let
  !l = reluLower eq
  !u = reluUpper eq
  !eqLength = length eq
  in if l > 0
    then eq ++ [0]
    else if u < 0
      then replicate (eqLength + 1) 0
    else
      composeLambdaAndNuRelu eq

applyRelu :: [[[Double]]] -> [[[Double]]]
applyRelu zonotope = let
  !zonotopeT = map Data.List.transpose zonotope
  !result = map (map applyReluPerDimension) zonotopeT
  !resultT = map Data.List.transpose result
  result' = if all (all (== 0) . last) resultT then map init resultT else resultT
  in result'
-- RELU TEST
test1Relu :: Bool
test1Relu = let
  expected = [[[2.025,2.025],
               [0.0,0.0],
               [1.8,1.8],
               [0.0,0.0],
               [0.0,0.0],
               [0.45,0.45],
               [0.22499999999999995,0.22499999999999995]],
               [[2.025,2.025],[0.0,0.0],[1.8,1.8],[0.0,0.0],[0.0,0.0],[0.45,0.45],[0.22499999999999995,0.22499999999999995]]] :: [[U.Vector Double]]
  x = [[2,2],
       [0,0],
       [2,2],
       [0,0],
       [0,0],
       [0.5,0.5]] :: [U.Vector Double]
  z = [x,x] :: [[U.Vector Double]]
  result = map (map U.fromList) (applyRelu (map (map U.toList) z))
  in result == expected

test2Relu :: Bool
test2Relu = let
  expected = [[[2,2],[0,0],[1,1],[0,0],[0,0],[0.5,0.5]],[[2,2],[0,0],[1,1],[0,0],[0,0],[0.5,0.5]]] :: [[U.Vector Double]]
  z = [[[2,2],[0,0],[1,1],[0,0],[0,0],[0.5,0.5]],[[2,2],[0,0],[1,1],[0,0],[0,0],[0.5,0.5]]] :: [[U.Vector Double]]
  result = map (map U.fromList) (applyRelu (map (map U.toList) z))
  in result == expected

-- PARSING THROUGH THE LAYERS
readLayers :: String -> IO [LayerInfo]
readLayers filepath = do
  !jsonData <- B.readFile filepath
  let !layers = fromMaybe [] (decode jsonData :: Maybe [LayerInfo])
  return layers

processLayers :: Handle -> String -> [[U.Vector Double]] -> IO [[U.Vector Double]]
processLayers logFile filepath zonotope = do
  !layers <- readLayers filepath
  case layers of
    [] -> do
      hPutStrLn logFile "Layers not found"
      return []
    layers' -> do
      let !inputShape' = fromMaybe (-1,-1) (inputShape (head layers'))
      case inputShape' of
        (-1,-1) -> do
          hPutStrLn logFile "Input shape for the neural network not found"
          return []
        inputShape'' -> do
          !finalZonotope <- parseLayers logFile (tail layers') zonotope inputShape''
          hPutStrLn logFile $ "Final zonotope dimensions: " ++ show (length finalZonotope,U.length (head (head finalZonotope)),length (head finalZonotope))
          return finalZonotope

{-
1 unique generator per dimension:
If zonotope is 3D and its center is (2,3,0.5), then there will be generators e1, e2 and e3 acting as follows:
2 + 1 e1
3 +       1 e2
0.5 +           1 e3
i.e
[[2,1,0,0],
 [3,0,1,0],
 [0.5,0,0,1]]
-}
convertImageDataToSingleZonotopePoint :: String -> IO ([[U.Vector Double]],Int)
convertImageDataToSingleZonotopePoint filepath = do
  !jsonData <- B.readFile filepath
  let !image = fromMaybe [] (decode jsonData :: Maybe [ImageData])
  case image of
    [] -> do
      putStrLn "Image data not found"
      return ([],-1)
    images' -> do
      let
        !img1 = imageValues (head images')
        -- !(height1,width1,_) = imageDimensions (head images')
        !concatImg = Data.List.transpose (concat img1)
        !img1Zonotope = map (\lst -> [U.fromList lst]) concatImg
      let !img1Class = imageClass (head images')
      return (img1Zonotope,img1Class)

parseLayers :: Handle -> [LayerInfo] -> [[U.Vector Double]] -> (Int,Int) -> IO [[U.Vector Double]]
parseLayers _ [] zonotope _ = return zonotope
parseLayers logFile (l:layers) zonotope (imgRows,imgCols) = do
  -- Print the layer type (name) of the current layer
  hPutStrLn logFile $ "parsed layer: " ++ layerName l
  hPutStrLn logFile $ "img dimensions: " ++ show (imgRows,imgCols)
  hPutStrLn logFile $ "zonotope dimensions: " ++ show (length zonotope,U.length (head (head zonotope)),length (head zonotope))
  case layerType l of
    "<class 'keras.src.layers.convolutional.conv2d.Conv2D'>" ->
      let
        !kernelSize' = fromMaybe [] (kernelSize l)
        !newRows = imgRows - head kernelSize' + 1
        !newCols = imgCols - head (tail kernelSize') + 1
        !newZ' = applyConvolution zonotope (imgRows,imgCols) (fromMaybe [] (filters l)) (fromMaybe [] (biases l))
        !activation = fromMaybe [] (activationFunction l)
        !newZ = if activation == "relu"
          then map (map U.fromList) (applyRelu (map (map U.toList) newZ'))
          else newZ'
      in parseLayers logFile layers newZ (newRows,newCols)
    "<class 'keras.src.layers.pooling.average_pooling2d.AveragePooling2D'>" ->
      let !poolSize' = fromMaybe [] (poolSize l)
          !p = head poolSize'
          !q = head (tail poolSize')
          !newRows = imgRows `div` p
          !newCols = imgCols `div` q
          !newZ = parseLayers logFile layers (applyAveragePooling p q zonotope (imgRows,imgCols)) (newRows,newCols)
      in newZ
    "<class 'keras.src.layers.reshaping.flatten.Flatten'>" ->
      let
        !newRows = 1
        !newCols = imgRows * imgCols
        !newZ' = map (map U.toList) zonotope
        !flattened = foldr (zipWith (++)) (repeat []) newZ'
        !flattenedVec = map U.fromList flattened
        !newZ = parseLayers logFile layers [flattenedVec] (newRows,newCols)
      in newZ
    "<class 'keras.src.layers.core.dense.Dense'>" ->
      let !weights' = fromMaybe [] (weights l)
          !weightsMatrix = M.transpose (M.fromLists weights')
          !zonotopeMatrix = M.transpose (M.fromLists (map U.toList (head zonotope)))
          !newZ2 = M.toLists (multStd weightsMatrix zonotopeMatrix)
          !biases' = fromMaybe [] (biases l)
          !newZ' = zipWith (\x row -> (head row + x) : tail row) biases' newZ2
          !newZ'' = Data.List.transpose newZ'
          !activationFunction' = fromMaybe [] (activationFunction l)
          !newZ = if activationFunction' == "relu"
            then map (map U.fromList) (applyRelu [newZ''])
            else [map U.fromList newZ'']
      in parseLayers logFile layers newZ (imgRows,imgCols)
    "<class 'keras.src.layers.regularization.dropout.Dropout'>" ->
      parseLayers logFile layers zonotope (imgRows,imgCols)
    _ ->
      return zonotope

-- CHECKING IF THE ARGMAX VALUE IN ALL POINTS MATCHES

checkArgMax :: [(Double,Double)] -> Int -> Bool
checkArgMax zonotope label = let
  !(lowerBoundForExpectedLabel,_) = zonotope !! label
  !otherTuples = take label zonotope ++ drop (label + 1) zonotope
  in all (\(_, y) -> y < lowerBoundForExpectedLabel) otherTuples

-- CREATING EQUATIONS FOR EACH DIMENSION (1 + 0 E1 + 1 E2 BECOMES [1,0,1])
{-
let zonotope = [1,2,3]
then equations = [(1 + 1 e1),
                  (2 +       1 e2),
                  (3 +             1 e3)]
i.e. equations = [1,1,0,0
                  2,0,1,0
                  3,0,0,1]
-}
createEquations :: U.Vector Double -> Double -> [U.Vector Double]
createEquations zonotope perturbation = let
  !size = U.length zonotope
  !identity' = M.identity size :: Matrix Double
  !perturbedIdentity = scaleMatrix perturbation identity'
  !perturbedIdentityVectors = map U.fromList (M.toLists perturbedIdentity)
  !newZ = zonotope : perturbedIdentityVectors
  in newZ

createHyperCubeEqs :: [[[Double]]] -> [[Double]]
createHyperCubeEqs eqs = let
  constants = head (head eqs) :: [Double]
  variables = tail (head eqs) :: [[Double]]
  variablesReduced = foldr (zipWith (+)) (repeat 0) variables :: [Double]
  in [constants,variablesReduced]

solveEquations :: [U.Vector Double] -> [(Double,Double)]
solveEquations zonotope = let
  !zonotopeLists = Data.List.transpose (map U.toList zonotope)
  !solved = map findBoundsPerDimension zonotopeLists
  in solved

findBoundsPerDimension :: [Double] -> (Double,Double)
findBoundsPerDimension equation = let
  !center = head equation
  !generators = tail equation
  !upperBound' = center + sum (map abs generators)
  !lowerBound' = center - sum (map abs generators)
  in (lowerBound',upperBound')

-- MAIN WITH ONLY A SINGLE POINT IN THE ZONOTOPE 
-- main :: IO ([[(Double, Double)]],[Bool])
-- main = do
--   logFile <- openFile "neuralNetVerification_output_memoryAnalysis.log" AppendMode
--   hSetBuffering logFile NoBuffering
--   (zonotope,correctLabel) <- convertImageDataToSingleZonotopePoint "/Users/prithvi/Documents/Krea/Capstone/AbstractVerification/Zonotope/haskell/app/imageData.json"
--   finalZonotopeEquations <- processLayers logFile "/Users/prithvi/Documents/Krea/Capstone/AbstractVerification/Zonotope/haskell/app/layersInfo.json" zonotope
--   let finalBounds = map solveEquations finalZonotopeEquations
--   let correctlyClassified = map (`checkArgMax` correctLabel) finalBounds
--   hPutStrLn logFile $ "Final zonotope" ++ show finalZonotopeEquations
--   hPutStrLn logFile "---------------------------------------------------"
--   hPutStrLn logFile $ "Final bounds" ++ show finalBounds
--   hPutStrLn logFile "---------------------------------------------------"
--   hPutStrLn logFile $ "Correctly classified: " ++ show correctlyClassified
--   hClose logFile
--   return (finalBounds,correctlyClassified)
-- readEquations :: String -> IO [[[Double]]]
-- readEquations filepath = do
--   !jsonData <- B.readFile filepath
--   let !equations = fromMaybe [] (decode jsonData :: Maybe [[[Double]]])
--   let hyperCubeEqs = createHyperCubeEqs equations
--   let hyperCubeEqsJson = encode hyperCubeEqs
--   B.writeFile "finalEquationsHyperCube.json" hyperCubeEqsJson
--   return equations

-- MAIN WITH ZONOTOPE WITH ALL EQUATIONS (ONE UNIQUE GENERATOR PER DIMENSION)
main :: IO ([[(Double,Double)]],[Bool])
main = do
  !logFile <- openFile "neuralNetVerification_output.log" AppendMode
  hSetBuffering logFile NoBuffering
  !args <- getArgs
  let !perturbation = read (head args) :: Double
  (!zonotope,!correctLabel) <- convertImageDataToSingleZonotopePoint "/data_home/Prithvi/haskell/app/imageData.json"
  let !perturbedZonotope = map (`createEquations` perturbation) (concat zonotope)
  !finalZonotopeEquations <- processLayers logFile "/data_home/Prithvi/haskell/app/layersInfo.json" perturbedZonotope
  let finalZonotopeEquationsJson = encode finalZonotopeEquations
  B.writeFile "finalEquationsAllVariables.json" finalZonotopeEquationsJson
  let finalZonotopeEquationsList = [map U.toList (head finalZonotopeEquations)] :: [[[Double]]]
  let hyperCubeEqs = createHyperCubeEqs finalZonotopeEquationsList
  let hyperCubeEqsJson = encode hyperCubeEqs
  B.writeFile "finalEquationsHyperCube.json" hyperCubeEqsJson
  let !finalBounds = map solveEquations finalZonotopeEquations
  let !correctlyClassified = map (`checkArgMax` correctLabel) finalBounds
  hPutStrLn logFile $ "Final zonotope classification: " ++ show correctlyClassified
  let finalBoundsJson = encode finalBounds
  B.writeFile "finalBounds.json" finalBoundsJson
  hClose logFile
  return (finalBounds,correctlyClassified)

-- "/data_home/Prithvi/haskell/app/imageData.json"
-- "/Users/prithvi/Documents/Krea/Capstone/AbstractVerification/Zonotope/haskell/app/imageData.json"
--  ghc -prof -fprof-auto -threaded -O2 -rtsopts MainEquations.hs  (While compiling)