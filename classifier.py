import numpy as np
import time

#-----------------------------------------------------------------------------------------------------------------------
#PERCEPTRON

#testing perceptron for a feature vector
def classify(w, fea):
    label = 0
    for i in range(len(fea)):
        if i in fea:
            label = label+w[i]*fea[i]
    if (label>=0):
        return 1
    else:
        return -1

#testing perceptron for a dataset, return error
def testPerceptron(w, feaSet, labelSet):
    nbErrs = 0
    for i in range(len(feaSet)):
        start = time.clock()
        labelPredicted = classify(w, feaSet[i])
        if (labelPredicted!=labelSet[i]):
            nbErrs = nbErrs+1
        print("Testing " + str(i) + " above " + str(len(feaSet)) + " : time = " + str(time.clock()-start))
    return (float(nbErrs)/len(feaSet))

#training perceptron
def trainPerceptron(trainFea, trainLabels, nbDim, learnRate, nbLoops):
    w     = [0] * nbDim
    nbSam = len(trainFea)
    err   = testPerceptron(w, trainFea, trainLabels)
    for it in range(nbLoops):
        print("Error on training : " + str(err) + ", iteration : " + str(it) + " over " + str(nbLoops))
        for i in range(nbSam):
            lp = classify(w, trainFea[i])
            if (lp!=trainLabels[i]):
                for k in range(nbDim):
                    if (k in trainFea[i]):
                        w[k] = w[k] + learnRate*trainLabels[i]*trainFea[i][k]
        err = testPerceptron(w, trainFea, trainLabels)
    print("Training finished")
    print("w = " + str(w))
    return w

#-----------------------------------------------------------------------------------------------------------------------
#BAYES CLASSIFIER

#calculate the probability of 0 and 1 of each dimension for each class
def probDim(feaVects, labels, nbDim):
    probs    = []
    probs.append([]); probs.append([]);
    probs[0] = [0] * nbDim #P(xo=1|C=spam)
    probs[1] = [0] * nbDim #P(xo=1|C=ham)
    nbSpam   = 0
    nbHam    = 0
    for i in range(len(feaVects)):
        if (labels[i]==-1):
            for k in range(nbDim):
                if (k in feaVects[i]):
                    probs[0][k] = probs[0][k]+1
            nbSpam = nbSpam+1
        else:
            for k in range(nbDim):
                if (k in feaVects[i]):
                    probs[1][k] = probs[1][k]+1
            nbHam = nbHam+1
    probs.append(float(nbSpam)/len(feaVects)) #P(C=spam)
    probs.append(float(nbHam)/len(feaVects))  #P(C=ham)
    return probs

#calculate the likelihood P(x|C=spam) and P(x|C=ham) based on the likelihood of single dimensions
def likelihoodSingle(fea, nbDim, probVect):
    likeli = []
    likeli.append(1.0); #P(x|C=spam)
    likeli.append(1.0); #P(x|C=ham)
    for i in range(nbDim):
        if (i in fea):
            likeli[0] = likeli[0]*probVect[0][i]
            likeli[1] = likeli[1]*probVect[1][i]
        else:
            likeli[0] = likeli[0]*(1-probVect[0][i])
            likeli[1] = likeli[1]*(1-probVect[1][i])
    return likeli

#calculate the likelihood P(x|C=spam) and P(x|C=ham) based on the likelihood of the whole vector
def likelihoodComposed(data, labels, fea, nbDim):
    cntSpam  = 0 ; cntHam  = 0 ;
    dataSpam = []; dataHam = [];
    for i in range(len(data)):
        if (labels[i]==1):
            dataHam.append(data[i])
        else:
            dataSpam.append(data[i])

    #P(x|C=spam)
    for i in range(len(dataSpam)):
        if (fea.keys()==dataSpam[i].keys()):
            cntSpam = cntSpam+1

    #P(x|C=ham)
    for i in range(len(dataHam)):
        if (fea.keys()==dataHam[i].keys()):
            cntHam = cntHam+1
    llh = [];
    #llh.append(float(cntSpam)/len(dataSpam)) ; #P(x|C=spam) without Laplace smoothing
    #llh.append(float(cntHam)/len(dataHam))   ; #P(x|C=ham) without Laplace smoothing
    llh.append(float(cntSpam+1)/(len(dataSpam)+nbDim))  ; #P(x|C=spam) with Laplace smoothing
    llh.append(float(cntHam+1)/(len(dataHam)+nbDim))    ; #P(x|C=ham) with Laplace smoothing
    llh.append(float(len(dataSpam))/len(data)); #P(C=spam)
    llh.append(float(len(dataHam))/len(data)) ; #P(C=ham)
    return llh

#calculate the probability P(C=spam|x) based on the probability of single dimensions
def probSpamS(fea, nbDim, probVect):
    P_xC = likelihoodSingle(fea, nbDim, probVect)
    P_x  = P_xC[0]*probVect[2] + P_xC[1]*probVect[3]
    P_Cx = P_xC[0]*probVect[2]/P_x
    return P_Cx

#calculate the probability P(C=spam|x) based on the probability of the whole vectors
def probSpamC(dataTrain, labelsTrain, fea, nbDimfea):
    P_xC = likelihoodComposed(dataTrain, labelsTrain, fea, nbDimfea)
    P_x  = P_xC[0]*P_xC[2] + P_xC[1]*P_xC[3]
    P_Cx = P_xC[0]*P_xC[2]/P_x
    return P_Cx

#testing Bayes using single dimensions probabilities
def testBayesS(data, labels, nbDim, probVect):
    labelPred = 0
    err       = 0
    for i in range(len(data)):
        pSpam = probSpamS(data[i], nbDim, probVect)
        if (pSpam>=0.5):
            labelPred = -1
        else:
            labelPred = 1
        if (labelPred!=labels[i]):
            err = err+1
    return (float(err)/len(data))

#testing Bayes using whole vector probabilities
def testBayesC(dataTest, labelsTest, dataTrain, labelsTrain, nbDim):
    labelPred = 0
    err       = 0
    for i in range(len(dataTest)):
        start = time.clock()
        pSpam = probSpamC(dataTrain, labelsTrain, dataTest[i], nbDim)
        if (pSpam>=0.5):
            labelPred = -1
        else:
            labelPred = 1
        if (labelPred!=labelsTest[i]):
            err = err+1
        print("Testing " + str(i) + " above " + str(len(dataTest)) + " : time = " + str(time.clock()-start))
    return (float(err)/len(dataTest))