import csv
import time
import numpy as np
import math
from io import StringIO 


def convertTextToNumpy(embeddingStr):
  embeddingStr = StringIO( unicode(embeddingStr) )
  return np.loadtxt(embeddingStr)

def mapThruDataSource(func):
  # func needs to return true or false to interact with the loop
  startTime = time.time()
  with open("../Data/Corpus/Glove/glove.6B.50d.txt") as csvFile:
    i = 0
    found = None
    
    while not found:
      lines = csvFile.readlines( 1024 *1024 )
      if len(lines) == 0:
        break
      
      if True:
        found = func(lines)
        i += 1
      if False:  
        for j, line in enumerate(lines):
          found = func(line)
          i += 1
          if found:
            splitLine = line.split(" ",1)
            break
    
      if i % 100 == 0 and False:
        lapsedTime = time.time() - startTime
        print("round {}, time lapsed {}".format(i,lapsedTime) )
        startTime = time.time()
    if found:
      print("found the word ", found[0] ," index ", i)
  if found:
    return convertTextToNumpy(found[1])
  else:
    return None

def findEmbedding(word):
  
  def checkLine(lines):
    for j, line in enumerate(lines):
      splitLine = line.split(" ",1)                
      if splitLine[0] == word:
        return splitLine
    
    return False
      
  return mapThruDataSource(checkLine)

def rankingList(word,indexList=[]):
  def compare(lines):
    if True:
      splitArray = []
      for j, line in enumerate(lines):
        splitArray.append(line.split(" ",1)[1])   
    
    comparedEmbedding = convertTextToNumpy("\n".join(splitArray))
    print(comparedEmbedding)
    exit()

    if False:
      for j, line in enumerate(lines):
        splitLine = line.split(" ",1)   
        comparedEmbedding = convertTextToNumpy(splitLine[1])
        result = calculateSimilarity(selectedWordEmbedding,comparedEmbedding,indexList)
        #print("selectedWordEmbedding",selectedWordEmbedding)
        #print("comparedEmbedding",comparedEmbedding)
        if result > 0.8:
        #if result < -0.5:
          print(result,splitLine[0])
    
  
  selectedWordEmbedding = findEmbedding(word)
  mapThruDataSource(compare)

def calculateSimilarity(word1Embedding, word2Embedding, indexList = []):
  return np.sum( np.multiply(normaliseVector(word1Embedding,indexList), normaliseVector(word2Embedding,indexList) ) )
    
def calculateSimilarityBetweenTwoWords(word1,word2, indexList = []):
  
  return calculateSimilarity( findEmbedding(word1) ,findEmbedding(word2) ,indexList)

def getFilterEmbedding(wordEmbedding,indexList=[]):
  filterEmbedding = np.ones_like(wordEmbedding)
  for i in indexList:
    filterEmbedding[i] = 0
  
  return filterEmbedding
  
def normaliseVector(wordEmbedding, indexList = []):
  
  # assume numbers in ndexList falls within wordEmbedding vector length
  #print("indexList noramliseVector",indexList)
  filterEmbedding = getFilterEmbedding(wordEmbedding,indexList)
  
  filteredWordEmbedding       = np.multiply(wordEmbedding,filterEmbedding)
  filteredWordEmbeddingSquare = np.square( filteredWordEmbedding )
  return np.divide(filteredWordEmbedding,np.sqrt( np.sum(filteredWordEmbeddingSquare) ) )

def calculateVectorRatioBetweenWords(word1,word2):
  
  def calculateVectorRatio(word1Embedding, word2Embedding):
    return np.divide(word1Embedding, word2Embedding)
    
  return calculateVectorRatio( findEmbedding(word1),  findEmbedding(word2) )  

def calculateMeanOfVector(wordEmbedding,indexList=[]):
    filterEmbedding = getFilterEmbedding(wordEmbedding, indexList)
    #~ print(filterEmbedding)
    count                 = np.sum(filterEmbedding)
    filteredWordEmbedding = np.multiply(wordEmbedding, filterEmbedding)
    meanOfVector          = np.sum(filteredWordEmbedding) / count
    #filteredMeanOfVector  = np.multiply(filterEmbedding,meanOfVector) 
    #return filteredMeanOfVector
    return meanOfVector

def calculateMeanOfWord(word,indexList=[]):
  return calculateMeanOfVector( findEmbedding(word) ,indexList)

# is this calculating correctly? is this formula for variance or standard deviation?
def calculateSTDEVofVector(wordEmbedding,indexList=[]):

  filterEmbedding       = getFilterEmbedding(wordEmbedding, indexList)
  filteredWordEmbedding = np.multiply(filterEmbedding,wordEmbedding)
  #print("filteredWordEmbedding", filteredWordEmbedding )
  mean  = calculateMeanOfVector(filteredWordEmbedding,indexList)
  count = np.sum(filterEmbedding) 
  if count < 2:
    return np.zeros(1,dtype=float)
  
  vectorOfSquareDifference = np.square( np.subtract(filteredWordEmbedding ,mean) )
  sumSquareDifference      = np.sum( vectorOfSquareDifference  )
  #~ print("filteredWordEmbedding",filteredWordEmbedding)
  #~ print("mean",mean)
  #~ print("np.subtract(filteredWordEmbedding ,mean)", np.subtract(filteredWordEmbedding ,mean))
  #~ print("vectorOfSquareDifference",vectorOfSquareDifference)
  #~ print("sumSquareDifference",sumSquareDifference)
  
  return np.sqrt( sumSquareDifference / (count - 1) )

def calculateRatioOfEmbedding(word1Embedding,word2Embedding):
  return np.divide(word1Embedding,word2Embedding)

# need a function to identify vector index within or beyond a specific standard deviation
# put in two word embedding, return list of index within or beyond specific standard deviation
def identifySimilarDimension(word1Embedding,word2Embedding,stdBound = 1,indexList =[]):
  resultEmbedding = calculateRatioOfEmbedding(word1Embedding,word2Embedding)
  resultStd  = calculateSTDEVofVector( resultEmbedding ,indexList)
  resultMean = calculateMeanOfVector(  resultEmbedding ,indexList)
  #print(" resultMean {} , resultStd {} ".format(resultMean,resultStd))
  #print("resultEmbedding",resultEmbedding)
  dissimilar = np.where( np.logical_or  (   resultEmbedding > (resultMean + resultStd * stdBound * 2 ) , 
                                            resultEmbedding < (resultMean - resultStd * stdBound * 2 ) ))
  similar = np.where( np.logical_and    (   resultEmbedding < (resultMean + resultStd * stdBound * 0.1 ) , 
                                            resultEmbedding > (resultMean - resultStd * stdBound * 0.1 ) ))
  return (similar, dissimilar)
  
  
#print( findEmbedding("sun") )
#print( findEmbedding("day") )
#print( normaliseVector( findEmbedding("day") ) )
#print( calculateSimilarityBetweenTwoWords("day","sun",[0,1])  )
#print( calculateVectorRatioBetweenWords( "day", "sun" ) )
#print( calculateVectorRatioBetweenWords( "day", "morning" ) )
#print( calculateMeanOfWord("day") )
#print( calculateSTDEVofVector(findEmbedding("day"),range(3,50)) )
#print( calculateRatioOfEmbedding(findEmbedding("day"), findEmbedding("sun")) )
#print( identifySimilarDimension(findEmbedding("day"), findEmbedding("sun")) )
#print( identifySimilarDimension(findEmbedding("monday"), findEmbedding("plane")) )
#print( identifySimilarDimension(findEmbedding("monday"), findEmbedding("day")) )
similarIndexList = identifySimilarDimension(findEmbedding("monday"), findEmbedding("gun"))
print("similarIndexList",similarIndexList[0])
print("similarIndexList dissimilar",similarIndexList[1])
print( rankingList("queen" ,[1,20,21]))
#print( rankingList("gun" ,similarIndexList[0]))

