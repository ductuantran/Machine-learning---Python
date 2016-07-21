import numpy as np
from collections import Counter

#extract all n-grams of a message
def ngramCalc(mess, n, sampleMess):
    ngrams     = []
    ngramsSamp = []
    longMess = ""
    #suppress 'enter' character and append lines into a long text
    subject = mess.subject
    content = mess.content
    for i in range(len(content)):
        content[i] = content[i][0:(len(content[i])-2)]
        longMess  += content[i]
    longMess = subject + " " + longMess
    keptSample = np.random.permutation(int(len(longMess)-n))[0:int(float(len(longMess)-n)*sampleMess)]
    for i in range(len(longMess)-n):
        ngrams.append(longMess[i:(i+n)])
    for i in range(len(keptSample)):
        ngramsSamp.append(ngrams[keptSample[i]])
    return ngramsSamp

#extract L most frequent n-grams of a corpus,
# where : sampleMess  is the proportion of n-grams kept for a message
#         sampleWhole is the proportion of n-grams kept the whole dataset
#(we do not take all the n-grams of a message due to its big volumn)
def mostFreqNgrams(data, L, n, sampleMess, sampleWhole):
    ngramsArr  = []
    for i in range(len(data)):
        ngrams = ngramCalc(data[i], n, sampleMess)
        if ((i%100)==0):
            print("Feature " + str(i) + " above " + str(len(data)) + " : vector size = " + str(len(ngrams)) + " x " + str(n))
        for k in range(len(ngrams)):
            ngramsArr.append(ngrams[k])
    print("Number of n-grams : " + str(len(ngramsArr)))

    #calculate frequencies
    print("\nCalculating frequency of occurences ...")
    #sample data
    idxSamples  = np.random.permutation(len(ngramsArr))[0:int(float(len(ngramsArr))*sampleWhole)]
    ngramsArrSp = []
    for i in range(len(idxSamples)):
        ngramsArrSp.append(ngramsArr[idxSamples[i]])
    c = Counter(ngramsArrSp)
    if (L>len(ngramsArrSp)):
        L = len(ngramsArrSp)
    LmostFreq = c.most_common(L)
    print("The " + str(L) + " most frequent n-grams : ")
    L = len(LmostFreq)
    for i in range(L):
        print("L = " + str(L) + ", n-grams : " + str(LmostFreq[i][0]) + ", appears : " + str(LmostFreq[i][1]) + " times")
    return LmostFreq

#extract L most frequent n-grams in both hams and spams
def mostFreqNgramsMix(data, L, n, sampleMess, sampleWhole):
    print("Finding the most discriminant n-grams ... ")
    dataSpam = []; dataHam = [];
    for i in range(len(data)):
        if (data[i].numLabel==1):
            dataHam.append(data[i])
        else:
            dataSpam.append(data[i])
    print("Finding the most frequent n-grams of the spams ... ")
    LmostSpam = mostFreqNgrams(dataSpam, L, n, sampleMess, sampleWhole)
    print("Finding the most frequent n-grams of the hams ... ")
    LmostHam  = mostFreqNgrams(dataHam, L, n, sampleMess, sampleWhole)
    LmostFreq = LmostSpam[0:int(float(L)/4)] + LmostSpam[int(float(L)*3/4):len(LmostSpam)] + LmostHam[0:int(float(L)/4)]
    LmostFreq = LmostFreq + LmostHam[(len(LmostHam)-(L-len(LmostFreq))):len(LmostHam)]
    return LmostFreq


#calculate feature of a mail
def feaCalc(mess, LmostFreq, sampleMess):
    fea = {}
    n = len(LmostFreq[0][0])
    ngrams = ngramCalc(mess, n, sampleMess)
    #feature  is a vector indicating the appearance of n-grams of the dictionary in a given message (both subject and content)
    for i in range(len(LmostFreq)):
        if (LmostFreq[i][0] in ngrams):
            fea[i] = 1
    return fea