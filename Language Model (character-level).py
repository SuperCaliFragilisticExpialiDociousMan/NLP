#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 18:56:13 2021

@author: shizhengyan
"""

#########################################
# Task1

import pandas as pd
import httpimport
import numpy as np

with httpimport.remote_repo(['lm_helper'], 'https://raw.githubusercontent.com/jasoriya/CS6120-PS2-support/master/utils/'):
  from lm_helper import get_train_data, get_test_data
  


# get the train and test data
train = get_train_data()
test, test_files = get_test_data()



from sklearn.feature_extraction.text import CountVectorizer
# formatting the doc to be used for training
print(train[1])
docList = []
for each in train:
    for l in each:
        docList.append(" ".join(l))


#print(docList[1])
#print(docList[2])
#print(docList[3])
#print(docList[4])




def getUnigramVals(docList):
    '''

    Generate a dictionary with unigrams as keys and their counts for given input as values
    e.g: {'a':53,'b':65}
    
    '''
    
    unigram_vectorizer = CountVectorizer(analyzer='char',ngram_range=(1,1),token_pattern=r'(?u)\b\w+\b')  #use this variable for defining CountVectorizer()
    unigram_vector = unigram_vectorizer.fit_transform(docList)  #use this variable for fit transform
    unigram_counts = np.sum(unigram_vector.toarray(),axis=0)   #use this counting unigrams
    D=dict(zip(unigram_vectorizer.get_feature_names(),unigram_counts))
    return D






def getBigramVals(docList):
    '''

    Generate a dictionary with unigrams as keys and their counts for given input as values
    e.g: {'ab':53,'bg':65}
    
    '''
    bigram_vectorizer = CountVectorizer(analyzer='char',ngram_range=(2,2)) 
    bigram_vector = bigram_vectorizer.fit_transform(docList)
    bigram_counts = np.sum(bigram_vector.toarray(),axis=0) 
    D=dict(zip(bigram_vectorizer.get_feature_names(),bigram_counts))
    return D


def getTrigramVals(docList):
    '''

    Generate a dictionary with unigrams as keys and their counts for given input as values
    e.g: {'abd':53,'bgv':65}
    
    '''
    trigram_vectorizer = CountVectorizer(analyzer='char',ngram_range=(3,3)) 
    trigram_vector = trigram_vectorizer.fit_transform(docList)
    trigram_counts = np.sum(trigram_vector.toarray(),axis=0) 
    D=dict(zip(trigram_vectorizer.get_feature_names(),trigram_counts))
    return D


#docList=[docList[1],docList[2],docList[3],docList[4]]


# Get the respective uni, bi and trigram values for the entire training data.
uniGram = getUnigramVals(docList)
biGram = getBigramVals(docList)
triGram = getTrigramVals(docList)
unigram_sum = sum(uniGram.values())

# Print the values of uni, tri and bigrams.
print("Length of unigram vocab:" + str(len(uniGram)))
print("Count of total unigrams:" + str(sum(uniGram.values())))
print("Length of bigram vocab:" + str(len(biGram)))
print("Count of total bigrams:" + str(sum(biGram.values())))
print("Length of trigram vocab:" + str(len(triGram)))
print("Count of total trigrams:" + str(sum(triGram.values())))


##############################################

# Task2
from sklearn.model_selection import train_test_split

import math

# split the training data into 80-20 ratio for finding lambdas  
docList_train, docList_test = train_test_split(docList, test_size=0.2) # use train_test_split on docList 
                               

# Respective uni, bi and trigrams for the 80 splitted training data
unigram_trained = getUnigramVals(docList_train)
bigram_trained = getBigramVals(docList_train)
trigram_trained = getTrigramVals(docList_train)
unigram_trained_sum = sum(unigram_trained.values())
bigram_trained_sum = sum(bigram_trained.values())
trigram_trained_sum = sum(trigram_trained.values())

# Uni, bi and trigram probability computation methods. 
def getUnigramProbability(l1, uniVals, unigramSum):
    if l1 not in uniVals:
        return 0
    else:
        return uniVals[l1]/unigramSum

def getBigramProbability(l1, l2, uniVals, biVals):
    if l1 not in uniVals or l1+l2 not in biVals:
        return 0
    else:
        return biVals[l1+l2]/uniVals[l1]
    
def getTrigramProbability(l1, l2, l3, triVals, biVals):
    if l1+l2 not in biVals or l1+l2+l3 not in triVals:
        return 0
    else:
        return triVals[l1+l2+l3]/biVals[l1+l2]

 

# compute the perplexity with linear interpolation   
def computePerplexityLinearInterpolation(doc, lambda1, lambda2, lambda3, unigram_trained, bigram_trained, trigram_trained, unigram_trained_sum):
    entropy = 0
    for i in range(0, len(doc)-2):
        l1 = doc[i]
        l2 = doc[i+1]
        l3 = doc[i+2]
        probability = lambda1*getUnigramProbability(l3, unigram_trained, unigram_trained_sum)+lambda2*getBigramProbability(l2, l3, unigram_trained, bigram_trained)+lambda3*getTrigramProbability(l1, l2, l3, trigram_trained, bigram_trained)
        if probability > 1:
            print(l1, l2, l3)
        if probability>0:
            entropy += math.log(probability)
    perplexity = 2**(-entropy/len(doc))
    return perplexity



# Make the held-out documents i.e books into a single stream or string of data as per the process followed in eisenstein for perplexity computation. Page: 139
held_out_doc = " ".join(docList_test)

lambdasToTry = [[0.1, 0.1, 0.8],[0.2,0.35,0.45],
                [0.25,0.3,0.45],[0.4,0.4,0.2],
                [0.2,0.4,0.4], [0.35,0.3,0.35],
                [0.5,0.25,0.25],[0.15,0.2,0.65],
                [0.6,0.3,0.1],[0.7,0.15,0.15],
                [0.1,0.5,0.4],[0.3,0.4,0.3]]
#  eg : [[0.1, 0.1, 0.8],[0.2,0.35,0.45],.........]]  



# list for held-out perplexities
perps = []
for a in lambdasToTry:
    lambda1, lambda2, lambda3 = a[0], a[1], a[2]
    perps.append(computePerplexityLinearInterpolation(held_out_doc, lambda1, lambda2, lambda3, unigram_trained, bigram_trained, trigram_trained,unigram_trained_sum))

print(perps)


# final lambdas based on minimum perplexities on held-out data
''' I found [0.1,0.1,0.8] can have a pretty small perplexities '''
finalLambdas = [0.1,0.1,0.8]
print(finalLambdas)



testDocList = []
for each in test:
    l=[]
    for i in each:
        l.append(" ".join(i))
    testDocList.append(" ".join(l))
#create the test data list in structured format

#list for storing perplexities of test docs
perplexities_test_docs_interpolation = []
for i in testDocList:
    perplexity=computePerplexityLinearInterpolation(i,0.7,0.15,0.15,unigram_trained, bigram_trained, trigram_trained,unigram_trained_sum)
    perplexities_test_docs_interpolation.append(perplexity)
#print(perplexities_test_docs_interpolation)

D=dict()
for i in range(len(testDocList)):
    D[testDocList[i]]=perplexities_test_docs_interpolation[i]

S = sorted(D.items(), key=lambda x:x[1])

#sort test docs based on perplexity on perplexities_test_docs_interpolation

#print all perplexities of docs in sorted order to find threshold

print(S)
#set the perplexity threshold based on observation after sorted values are printed.
'''I would say the threshold is above 6.371696815776572'''
perplexity_threshold_for_interpolation = 6.371696815776572


#Classify non english docs based on perplexity threshold and printing their filenames along with perplexity for manual validation on perplexities_test_docs_interpolation

test_filename_List = []
for i in test_files:
    test_filename_List.append(i)
        
D2=dict()
for i in range(len(testDocList)):
    D2[testDocList[i]]=[perplexities_test_docs_interpolation[i],
                       test_filename_List[i]]

S2 = sorted(D2.items(), key=lambda x:x[1])
S3=S2
for i in S3:
    if i[1][0]<=6.371696815776572:
        S3.remove(i)
outcome=[]
for i in S3:
    outcome.append(i[1])
print(outcome)




#print few lines from the doc, perplexity value, filename and classification by the model by using perplexities_test_docs_interpolation
demo=[]
for i in S2:
    l=[]
    l.append(i[0][:100])
    l.append(i[1])
    if i[1][0]<=6.371696815776572:
        l.append('English')
    else:
        l.append('Non-English')
    demo.append(l)
print(demo)










######################################################

# Task3

# Method for calculating trigram probability 
def getTrigramProbabilityWithLambdaSmoothing(l1, l2, l3, triVals, biVals, lambdaVal): 
    if l1+l2+l3 not in triVals and l2+l3 not in biVals:
        return 1/len(biVals)
    elif l1+l2+l3 not in triVals and l2+l3 in biVals:
        return lambdaVal / (biVals[l2+l3]+lambdaVal*len(biVals))
    elif l1+l2+l3 in triVals and l2+l3 not in biVals:
        return (triVals[l1+l2+l3]+lambdaVal) / (lambdaVal*len(biVals))
    return (triVals[l1+l2+l3]+lambdaVal) / (biVals[l2+l3]+lambdaVal*len(biVals))
            

# Method for perplexity calculation with lambda smoothing
def computePerplexityWithLambdaSmoothing(doc, biVals, triVals):
    entropy = 0
    for i in range(len(doc)-2):
        l1 = doc[i]
        l2 = doc[i+1]
        l3 = doc[i+2]
        probability = getTrigramProbabilityWithLambdaSmoothing(l1, l2, l3, triVals, biVals, 0.1)
        entropy  += math.log(probability)
    perplexity = 2**(-entropy/len(doc))
    return perplexity

#Testing lambda smoothened trigram perplexity on held-out data


# list for perplexities and details in test data
perplexities_test_docs_lambda_smoothing = []
for each in testDocList:
    perplexity=computePerplexityWithLambdaSmoothing(each, bigram_trained, trigram_trained)
    perplexities_test_docs_lambda_smoothing.append(perplexity)

print(perplexities_test_docs_lambda_smoothing)




#sort perplexities_test_docs_lambda_smoothing based on perplexity 
d=dict()
for i in range(len(testDocList)):
    d[testDocList[i]]=perplexities_test_docs_lambda_smoothing[i]

s = sorted(d.items(), key=lambda x:x[1])

#print all the perplexities in sorted order to figure out threshold
print(s)

#perplexity threshold set based on observation after sorted values are printed.
perplexity_threshold_for_lambda_smoothing =12.184154739880812 #select value here



#Classify non english docs based on perplexity threshold and print their filenames along with perplexity for manual validation on perplexities_test_docs_lambda_smoothing
test_filename_List = []
for i in test_files:
    test_filename_List.append(i)
        
d2=dict()
for i in range(len(testDocList)):
    d2[testDocList[i]]=[perplexities_test_docs_lambda_smoothing[i],
                       test_filename_List[i]]

s2 = sorted(d2.items(), key=lambda x:x[1])
s3=s2
for i in s3:
    if i[1][0]<=12.184154739880812:
        s3.remove(i)
outcome=[]
for i in s3:
    outcome.append(i[1])
print(outcome)



#print few lines from the doc, perplexity value, filename and classification by the model perplexities_test_docs_lambda_smoothing.
demo=[]
for i in S2:
    l=[]
    l.append(i[0][:100])
    l.append(i[1])
    if i[1][0]<=6.371696815776572:
        l.append('English')
    else:
        l.append('Non-English')
    demo.append(l)
print(demo)





