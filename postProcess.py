import numpy as np
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from scipy.spatial import distance

#feature selection by Pearson Correlation
def calPearson(fea, label, nbDim):
    pearArr = []
    for i in range(nbDim):
        mess = [0] * len(fea)
        for k in range(len(fea)):
            if (i in fea[k]):
                mess[k] = fea[k][i]
        pearArr.append(abs(pearsonr(mess, label)[0]))
    return pearArr

#feature selection by Mutual Information
#fea   : data under the form of a dictionary (sparse array)
#label : labels of data
#nbDim : dimension of a feature vector
def calcMI(fea, label, nbDim):
    MIArr = []
    #calculate MI for each dimension (feature)
    nbSam = len(fea)
    print("Number of samples : " + str(nbSam) + ", size of feature vector : " + str(nbDim))
    print("Calculating mutual information (MI) ... ")
    for i in range(nbDim):
        pf0  = 0 #probability for fea = 0
        pl0  = 0 #probability for label = -1
        pf1  = 0 #probability for fea = 1
        pl1  = 0 #probability for label = 1
        #joint probabilities
        pfl1 = 0; pfl2 = 0; pfl3 = 0; pfl4 = 0;

        #set 1: (dim i = 1, label = 1)
        #set 2: (dim i = 0, label = -1)
        #set 3: (dim i = 0, label = 1)
        #set 4: (dim i = 1, label = -1)
        for k in range(nbSam):
            if (i in fea[k]):
                pf1 = pf1+1
            else:
                pf0 = pf0+1
            if (label[k]==1):
                pl1 = pl1+1
            else:
                pl0 = pl0+1
            if ((i in fea[k])&(label[k]==1)):
                pfl1 = pfl1+1
            if ((i not in fea[k])&(label[k]==-1)):
                pfl2 = pfl2+1
            if ((i not in fea[k])&(label[k]==1)):
                pfl3 = pfl3+1
            if ((i in fea[k])&(label[k]==-1)):
                pfl4 = pfl4+1
        #mutual information
        pf1   = float(pf1)/nbSam ; pl1  = float(pl1)/nbSam ; pf0  = float(pf0)/nbSam ; pl0  = float(pl0)/nbSam ;
        pfl1  = float(pfl1)/nbSam; pfl2 = float(pfl2)/nbSam; pfl3 = float(pfl3)/nbSam; pfl4 = float(pfl4)/nbSam;
        MI1   = 0;  MI2 = 0; MI3 = 0; MI4 = 0;
        if ((pfl1*pf1*pl1)!=0):
            MI1 = pfl1*np.log(pfl1/(pf1*pl1))
        if ((pfl2*pf0*pl0)!=0):
            MI2 = pfl2*np.log(pfl2/(pf0*pl0))
        if ((pfl3*pf0*pl1)!=0):
            MI3 = pfl3*np.log(pfl3/(pf0*pl1))
        if ((pfl4*pf1*pl0)!=0):
            MI4 = pfl4*np.log(pfl4/(pf1*pl0))
        #MI = abs(MI1) + abs(MI2) + abs(MI3) + abs(MI4) #MI1
        MI = MI1 + MI2 + MI3 + MI4 #MI2
        #MI = max(pfl1/pf1+pfl2/pf0,pfl3/pf0+pfl4/pf1)   #MI3
        MIArr.append(MI)
        print("Dim " + str(i) + " : MI = " + str(MI))
    return MIArr

#find out dimensions to reduce
def findDimToReduce(fea, label, nbDimIn, nbDimOut, criterion):
    print("Finding dimensions to reduce ...")
    feaOut    = []
    metricArr = []
    if (criterion=="MI"):
        print("Using Mutual Information")
        metricArr = calcMI(fea, label, nbDimIn)
    elif (criterion=="Pearson"):
        print("Using Pearson criterion")
        metricArr = calPearson(fea, label, nbDimIn)
    maxIdx = list(reversed(sorted(range(len(metricArr)), key=lambda k: metricArr[k])))
    idx    = maxIdx[0:nbDimOut]
    return idx

#reduce dimension of data using feature selection
def reduceByFs(fea, label, nbDimIn, nbDimOut, dimsToReduce):
    print("\n\nDimensional reduction ... ")
    feaOut = []
    for i in range(len(fea)):
        if ((i%100)==0):
            print("Dimensional reduction : " + str(i) + " above " + str(len(fea)))
        feaR = {}
        for k in range(nbDimOut):
            if (dimsToReduce[k] in fea[i]):
                feaR[k] = fea[i][dimsToReduce[k]]
        feaOut.append(feaR)
    return feaOut

#find centers by K-means
def findKmeansCenters(fea, nbDim, nbClusters):
    print("Finding clusters of data by K-means ...")
    maxIter = 300
    feaVect = []
    for i in range(len(fea)):
        feaMess = [0] * nbDim
        for k in range(nbDim):
            if (k in fea[i]):
                feaMess[k] = fea[i][k]
        feaVect.append(feaMess)
    clusters = KMeans(n_clusters=nbClusters,max_iter=maxIter).fit(feaVect)
    centers  = clusters.cluster_centers_
    return centers

#reduce dimension of data using K-means and Bag-of-Words
def reduceByKmeans(fea, labels, nbDim, centers):
    #dimensional reduction by calculating distances from a feature vector to all the centers to form the new feature vector
    feaVect = []
    for i in range(len(fea)):
        feaMess = [0] * nbDim
        for k in range(nbDim):
            if (k in fea[i]):
                feaMess[k] = fea[i][k]
        feaVect.append(feaMess)
    print("Dimensional reduction by K-means ...")
    feaOut = []
    for i in range(len(feaVect)):
        feaMess = {}
        for k in range(len(centers)):
            feaMess[k] = distance.euclidean(feaVect[i],centers[k])
        if ((i%100)==0):
            print("Dimensional reduction " + str(i) + " above " + str(len(feaVect)) + ". Label = " + str(labels[i]) + ". Vector = " + str(feaMess))
        feaOut.append(feaMess)
    return feaOut

#count number of appearances of a dimension in spam and ham
def countAppOfDim(fea, nbDim, label):
    print("Counting contribution of a dimension in ham/spam ...")
    for i in range(nbDim):
        h1s0 = 0; h0s1 = 0; feaBin = 0;
        for k in range(len(fea)):
            if i in fea[k]:
                feaBin = 1
            else:
                feaBin = 0
            if (((label[k]==1)and(feaBin==1))or((label[k]==-1)and(feaBin==0))):
                h1s0 = h1s0+1
            if (((label[k]==1)and(feaBin==0))or((label[k]==-1)and(feaBin==1))):
                h0s1 = h0s1+1
        h1s0 = float(h1s0)/len(fea)
        h0s1 = float(h0s1)/len(fea)
        print("Dim " + str(i) + " : h1s0 = " + str(h1s0) + ", h0s1 = " + str(h0s1))


