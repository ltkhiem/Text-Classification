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

print (trainList.__len__())

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

for N0 in TRAIN_NUM:
	while len(sampleSet) < N0:
		ID = random.randrange(len(trainList))
		sampleSet.append(trainList[ID])
		del trainList[ID]

	# #Dictionary Building
	# print("Building dictionary")
	# vocabulary = []
	# for doc in sampleSet:
	# 	f = open(doc, "r")
	# 	words = f.read().split()
	# 	wordsSet.update(words)
	# 	f.close()

	# vocabulary = random.sample(wordsSet, 2000)
	# f = open("vocabulary.pickle","wb")
	# pickle.dump(vocabulary,f)
	# f.close()

	#Inverted Index Building
	print("Building inverted index")
	invertedIndex = {}
	for term in vocabulary:
		invertedIndex[term] = {}

	print("Caclculating Inverted Index")
	for doc in sampleSet:
		print(doc)
		f = open(doc,"r")
		words = Counter(f.read().split())
		for word in words:
			if invertedIndex.get(word)!=None:
				invertedIndex[word][doc] = words[word]
		f = open("KNN_inverted_index_"+str(N0)+".pickle","wb")
		pickle.dump(invertedIndex,f)
		f.close()

	#TF-IDF Calculating
	print("Calculating TF-IDF")
	weight = {}
	for doc in sampleSet:
		tmp = np.zeros(vocabulary.__len__())
		for i,term in enumerate(vocabulary):
			if invertedIndex[term].get(doc)!=None:
				tf = 1 + math.log(invertedIndex[term][doc])
			else:
				tf = 0
			idf = math.log(len(sampleSet)/(1+len(invertedIndex[term])))
			tmp[i] = tf*idf
		weight[doc] = UnitVector(tmp)

	if TEST_KNN :
		category = {}
		count = 0

		for doc in testList:
			testWeight = np.zeros(vocabulary.__len__())
			compareSet = set()
			f = open(doc, "r")
			words = Counter(f.read().split())
			for i,word in enumerate(vocabulary):
				if (words.get(word)!=None):
					tf = 1 + math.log(words[word])
					idf = math.log(len(sampleSet)/(1+len(invertedIndex[word])))
					testWeight[i] = tf*idf
					compareSet.update(list(invertedIndex[word]))

			f.close()

			testWeight = UnitVector(testWeight)
			rank = []
			for sample in compareSet:
				rank.append([Angle(testWeight,weight[sample]), sample])
			rank = sorted(rank)

			temp = {}
			for i in range(min(K, rank.__len__())):
				cat = str(GetType(rank[i][1]))
				if (temp.get(cat)!=None):
					temp[cat] += 1
				else:
					temp[cat] = 1
			max = -1
			category[doc] = "None"
			for item in temp:
				if temp[item] > max:
					max = temp[item]
					category[doc] = item
			count += 1
			print (count, doc, category[doc], max)

			f = open("KNN_category_"+str(N0)+".pickle","wb")
			pickle.dump(weight,f)
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
	f = open("KNN_Result_"+str(N0)+".txt", "w")
	f.write(str(truePositive/5000))
	f.close()


