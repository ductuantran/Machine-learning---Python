from os import listdir
from os.path import isfile, join
import msvcrt
import copy
import numpy as np

#message structure
class message:
    link     = ""
    subject  = []
    content  = []
    label    = []
    numLabel = []

#data structure
class dataStruct:
    train = []
    test  = []

#find positions of a character in a string
def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

#read data from a file
def readData(fileName):
    with open(fileName) as f:
        content = f.readlines()
    firstLine = content[0]
    mess = message()
    mess.subject = firstLine[9:]
    mess.content = content[1:]
    pos = find(fileName, ".")
    mess.label = fileName[(pos[len(pos)-2]+1):(pos[len(pos)-1])]
    if (mess.label=="ham"):
        mess.numLabel = 1
    else:
        mess.numLabel = -1
    mess.link    = fileName
    return mess

#read data from the whole dataset, trainRate is the proportion of the database reserved for training
def readWholeData(folderName, trainRate, testRate):
    print("Reading data ...")
    data = dataStruct()
    for i in range(1,7,1):
        subFolder = folderName + "/enron" + str(i)
        #read hams and divide into training and testing sets
        subHam     = subFolder + "/ham"
        hamFiles   = [f for f in listdir(subHam) if isfile(join(subHam, f))]
        rdHam      = np.random.permutation(len(hamFiles))
        rdHamTrain = rdHam[0:int(trainRate*len(rdHam))]
        rdHamTest  = rdHam[int(trainRate*len(rdHam)):(int(trainRate*len(rdHam))+int(testRate*len(rdHam)))]
        for k in range(len(hamFiles)):
            fileName = subHam + "/" + str(hamFiles[k])
            if (k in rdHamTrain):
                data.train.append(readData(fileName))
            elif (k in rdHamTest):
                data.test.append(readData(fileName))

        #read spams and divide into training and testing sets
        subSpam     = subFolder + "/spam"
        spamFiles   = [f for f in listdir(subSpam) if isfile(join(subSpam, f))]
        rdSpam      = np.random.permutation(len(spamFiles))
        rdSpamTrain = rdSpam[0:int(trainRate*len(rdSpam))]
        rdSpamTest  = rdSpam[int(trainRate*len(rdSpam)):(int(trainRate*len(rdSpam))+int(testRate*len(rdSpam)))]
        for k in range(len(spamFiles)):
            fileName = subSpam + "/" + str(spamFiles[k])
            if (k in rdSpamTrain):
                data.train.append(readData(fileName))
            elif (k in rdSpamTest):
                data.test.append(readData(fileName))
    print("Data reading finished")
    return data