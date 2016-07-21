import numpy as np
import ioData as io
import featureCalc as fea
import postProcess as pp
import classifier as cl

#parameters
trainRate   = 0.7  #proportion of the database which is used as training set
testRate    = 0.3  #proportion of the database which is used as testing set
sampleMess  = 0.7  #proportion of the total number of the n-grams of a message which are kept for use
sampleWhole = 0.7  #proportion of the total number of training samples (n-grams) which are kept for use
nbWords     = 50   #number of most frequent n-grams (number of words in the dictionary)
ngramLen    = 3    #length of an n-gram (n)

#main program

#read all data
folderName = "Dataset"
data = io.readWholeData(folderName, trainRate, testRate)
print("Number of training data samples : " + str(len(data.train)))
print("Number of testing data samples : " + str(len(data.test)))
print("Total number of data samples : " + str(len(data.train)+len(data.test)))

#calculate dictionary
LmostFreq   = fea.mostFreqNgrams(data.train, nbWords, ngramLen, sampleMess, sampleWhole)
#LmostFreq = fea.mostFreqNgramsMix(data.train, nbWords, ngramLen, sampleMess, sampleWhole)

#calculate features by dictionary
feaTrain = [] ; labelTrain = []
feaTest  = [] ; labelTest  = []
print("\n\nCalculating feature of all the training set ...")
for i in range(len(data.train)):
    if ((i%100)==0):
        print("Feature " + str(i) + " above " + str(len(data.train)))
    feaMess = fea.feaCalc(data.train[i], LmostFreq, sampleMess)
    feaTrain.append(feaMess)
    labelTrain.append(data.train[i].numLabel)
print("Calculation of training feature finished")

print("\n\nCalculating feature of all the testing set ...")
for i in range(len(data.test)):
    if ((i%100)==0):
        print("Feature " + str(i) + " above " + str(len(data.test)))
    feaMess = fea.feaCalc(data.test[i], LmostFreq, sampleMess)
    feaTest.append(feaMess)
    labelTest.append(data.test[i].numLabel)
print("Calculation of testing feature finished")


#dimensional reduction by feature selection
print("\n\nFeature selection ...")
criterion       = "Pearson"
nbWords         = len(LmostFreq)
nbDimOut        = int(float(nbWords)/5)
dimsToReduce    = pp.findDimToReduce(feaTrain, labelTrain, nbWords, nbDimOut, criterion)
feaTrainReduced = pp.reduceByFs(feaTrain, labelTrain, nbWords, nbDimOut, dimsToReduce)
feaTestReduced  = pp.reduceByFs(feaTest, labelTest, nbWords, nbDimOut, dimsToReduce)

#dimensional reduction by Kmeans
#print("\n\nDimensional reduction using K-means clustering ...")
#nbClusters      = 20
#nbDimOut        = nbClusters
#centers         = pp.findKmeansCenters(feaTrain, nbWords, nbClusters)
#feaTrainReduced = pp.reduceByKmeans(feaTrain, labelTrain, nbWords, centers)
#feaTestReduced  = pp.reduceByKmeans(feaTest, labelTrain, nbWords, centers)

#training perception
print("\n\nTraining perceptron ...")
w         = []
learnRate = 0.001
nbLoops   = 300
w         = cl.trainPerceptron(feaTrain, labelTrain, nbWords, learnRate, nbLoops)
errOnTest = cl.testPerceptron(w, feaTest, labelTest)
print("Size of vectors : " + str(nbWords) + ". Error on testing : "+ str(errOnTest))

#Bayes classifier
print("\n\nClassification by Bayes ...")
print("Calculating probability of a value for each dimension ...")
probVect = cl.probDim(feaTrain, labelTrain, nbWords)
print("Testing Bayes classifier ... ")
bayesErrTrain = cl.testBayesC(feaTrain, labelTrain, feaTrain, labelTrain, nbWords)
bayesErrTest  = cl.testBayesC(feaTest, labelTest, feaTrain, labelTrain, nbWords)
print("Error of Bayes classifier on the training set : " + str(bayesErrTrain))
print("Error of Bayes classifier on the testing set : " + str(bayesErrTest))