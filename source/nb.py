from configuration import *
from collections import Counter
import random
import os
import pickle
import math
import re
import numpy as np

NEWSAMPLE = 0
TEST_KNN = 1
EPS = 1e-9
K = 5

random.seed(64)

docList = []
for dirs, subdirs, files in os.walk(DATASET_DIR):
	for file in files:
		docList.append(os.path.join(dirs,file))

trainList = docList[:]
testList = []
count = 0
while (count < TEST_DATA_SIZE) :
	ID = random.randrange(len(trainList))
	testList.append(trainList[ID])
	del trainList[ID]
	count += 1

TRAIN_NUM = [1000, 3000, 5000, 7000, 9000, 11000, 13000, 14997]
sampleSet = []
wordsSet = set()

def UnitVector(vector):
    vector = np.array(vector)
    if (np.linalg.norm(vector))==0:
        return None 
    return vector/ np.linalg.norm(vector)

def Angle(a, b):
    if (a is None) or (b is None):
        return 9999999
    return np.arccos(np.clip(np.dot(a,b), -1.0,1.0))	

def GetType(doc):
	doc = str(doc)
	return re.search("../dataset/20_newsgroups/(.*)\\\\.*", doc).group(1)


f = open("vocabulary.pickle","rb")
vocab = pickle.load(f)
vocabulary = list(vocab)
f.close()

C = os.listdir(DATASET_DIR)

def GetIndex(type):
	return C.index(type)

for N0 in TRAIN_NUM:
	#Training
	print("Training")
	while len(sampleSet) < N0:
		ID = random.randrange(len(trainList))
		sampleSet.append(trainList[ID])
		del trainList[ID]
	D = sampleSet.__len__()	
	DI = [0.0 for i in range(C.__len__())]
	PC = [0.0 for i in range(C.__len__())]
	TI = [Counter([]) for i in range(C.__len__())]
	NI = [0 for i in range(C.__len__())]
	for sample in sampleSet:
		type = GetType(sample)
		index = GetIndex(type)
		DI[index] += 1

		f = open(sample,"r")
		words = Counter(f.read().split())
		TI[index] += words
		
	for i in range(C.__len__()):
		PC[i] = DI[i]/D
		for term in vocabulary:
			NI[i] += TI[i][term]

	P = {} 
	for term in vocabulary:
		P[term] = {}
		for i in range(C.__len__()):
			nij = TI[i][term]
			P[term][i] = (nij+1) / (NI[i]+vocabulary.__len__())

	
	#Testing
	print("Testing")
	count = 0
	category = {}
	for doc in testList:
		category[doc] = "None"
		f = open(doc,"r")
		words = Counter(f.read().split())
		max = -1
		for i in range(C.__len__()):
			res = 1
			for word in vocabulary:
				if (word in words):
					res *= words[word]*P[word][i]
			res*=PC[i]
			if (res>max):
				max = res
				category[doc] = C[i]
		count+=1
		print(count, doc, category[doc])

	f = open("NB_category_"+str(N0)+".pickle","wb")
	pickle.dump(category,f)
	f.close()

	#Evaluation
	truePositive = 0
	numUnknown = 0
	for i in category:
		type = GetType(i)
		if (type==category[i]):
			truePositive += 1
	print("----------------------------------------------------------------------------------")
	print("Accuracy : ", truePositive)
	print("----------------------------------------------------------------------------------")
	f = open("NB_Result_"+str(N0)+".txt", "w")
	f.write(str(truePositive/5000))
	f.close()
