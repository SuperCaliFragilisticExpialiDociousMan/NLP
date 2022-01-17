#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 15:24:29 2021

@author: shizhengyan
"""


'Cross-Language Word Embeddings '

import gensim
import numpy as np
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

lines = [s.split() for s in open('/users/shizhengyan/Desktop/Neu/CS6120/hw4/shakespeare_plays.txt')]

model = Word2Vec(lines,min_count=1,size=50,workers=3,window=3,sg=1)
# size – Dimensionality of the word vectors
# window – Maximum distance between the current and predicted word within a sentence
# min_count – Ignores all words with total frequency lower than this
# workers – Use these many worker threads to train the model (=faster training with multicore machines)
# sg - 1 for skip-gram; otherwise CBOW

print(model.wv.most_similar(positive='husband', negative='majesty'))

print(model.wv.most_similar(positive='othello'))

print('similarity between othello and desdemona: ',model.wv.similarity('othello', 'desdemona'))



##############################################################
'Implement Cosim'

def cosim(v1, v2): 
    ### START CODE HERE ###
    # Compute the dot product between u and v (≈1 line)
    dot = np.dot(v1,v2)
    # Compute the L2 norm of u (≈1 line)
    norm_v1 = np.linalg.norm(v1)
    
    # Compute the L2 norm of v (≈1 line)
    norm_v2 = np.linalg.norm(v2)
    # Compute the cosine similarity defined by formula (1) (≈1 line)
    cosine_similarity = dot/(norm_v1*norm_v2)

    return cosine_similarity

print('similarity between othello and desdemona: ',cosim(model.wv['othello'], model.wv['desdemona']))
print('model.wv[othello]: ',model.wv['othello'])



def vecref(s):
    (word, srec) = s.split(' ', 1)
    return (word, np.fromstring(srec, sep=' '))

def ftvectors(fname):
    return { k:v for (k, v) in [vecref(s) for s in open(fname)] if len(v) > 1} 

# loading vectors for english and french languages.
envec = ftvectors('/users/shizhengyan/Desktop/Neu/CS6120/hw4/30k.en.vec')
frvec = ftvectors('/users/shizhengyan/Desktop/Neu/CS6120/hw4/30k.fr.vec')
#print('envec: ',envec)

# TODO: load vectors for one more language, such as zhvec (Chinese) just like english or french


##################################################################
' Implement search'
## TODO: implement this search function
def mostSimilar(vec, vecDict):
  ## Use cosim function from above
    mostSimilar = ''
    similarity = 0
    for row in vecDict.items():
        csm = cosim(vec, row[1])
        if csm>similarity:
            similarity = csm
            mostSimilar = row[0]
    return (mostSimilar, similarity)

## some example searches
print('mostSimilar(envec[e], frvec) for e in ***: ',[mostSimilar(envec[e], frvec) for e in ['computer', 'germany', 'matrix', 'physics', 'yeast']])

links = [s.split() for s in open('/users/shizhengyan/Desktop/Neu/CS6120/hw4/links.tab')][0:999]

print('len of links: ',len(links))
print('links[302]: ',links[302])

############################################################

'Evaluate embeddings'
## TODO: Compute English-French Wikipedia retrieval accuracy.
t = 0
b = 0
g = 0
num=1
for row in links:
    num+=1
    if num%1000==0:
        print(num)
    if row[1] == 'fr':
        if row[0] in envec.keys():
            t += 1
            if row[0] == row[2]:
                b += 1
            similar, _ = mostSimilar(envec[row[0]], frvec)
            if similar==row[2]:
                g += 1

baselineAccuracy = b/t
accuracy = g/t

print(baselineAccuracy, accuracy)
# 0.6742324450298915 0.5359205593271862



#######################################

'compute accuracy'


## TODO: Compute English-X Wikipedia retrieval accuracy.
#Follow the above procedure to do this task.


devec = ftvectors('/users/shizhengyan/Desktop/Neu/CS6120/hw4/30k.de.vec')

links = [s.split() for s in open('/users/shizhengyan/Desktop/Neu/CS6120/hw4/links.tab')][0:999]

t = 0
b = 0
g = 0
num=1
num_ar=0
for row in links:
    num+=1
    if num%1000==0:
        print(num)
    if row[1] == 'de':
        num_ar+=1
        if row[0] in envec.keys():
            t += 1
            if row[0] == row[2]:
                b += 1
            similar, _ = mostSimilar(envec[row[0]], devec)
            if similar==row[2]:
                g += 1

baselineAccuracy = b/t
accuracy = g/t

print(baselineAccuracy, accuracy)









