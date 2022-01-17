#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 15:24:29 2021

@author: shizhengyan
"""


import pandas as pd
from collections import defaultdict
import math
import numpy as np
import string


# Punctuation characters
punct = set(string.punctuation)

# Morphology rules used to assign unknown word tokens
noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
verb_suffix = ["ate", "ify", "ise", "ize"]
adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
adv_suffix = ["ward", "wards", "wise"]


def get_word_tag(line, vocab): 
    if not line.split():
        word = "--n--"
        tag = "--s--"
        return word, tag
    else:
        word, tag = line.split()
        if word not in vocab: 
            # Handle unknown words
            word = assign_unk(word)
        return word, tag
    return None 


def preprocess(vocab, data_fp):
    """
    Preprocess data
    """
    orig = []
    prep = []

    # Read data
    with open(data_fp, "r") as data_file:

        for cnt, word in enumerate(data_file):

            # End of sentence
            if not word.split():
                orig.append(word.strip())
                word = "--n--"
                prep.append(word)
                continue

            # Handle unknown words
            elif word.strip() not in vocab:
                orig.append(word.strip())
                word = assign_unk(word)
                prep.append(word)
                continue

            else:
                orig.append(word.strip())
                prep.append(word.strip())

    assert(len(orig) == len(open(data_fp, "r").readlines()))
    assert(len(prep) == len(open(data_fp, "r").readlines()))

    return orig, prep


def assign_unk(tok):
    """
    Assign unknown word tokens
    """
    # Digits
    if any(char.isdigit() for char in tok):
        return "--unk_digit--"

    # Punctuation
    elif any(char in punct for char in tok):
        return "--unk_punct--"

    # Upper-case
    elif any(char.isupper() for char in tok):
        return "--unk_upper--"

    # Nouns
    elif any(tok.endswith(suffix) for suffix in noun_suffix):
        return "--unk_noun--"

    # Verbs
    elif any(tok.endswith(suffix) for suffix in verb_suffix):
        return "--unk_verb--"

    # Adjectives
    elif any(tok.endswith(suffix) for suffix in adj_suffix):
        return "--unk_adj--"

    # Adverbs
    elif any(tok.endswith(suffix) for suffix in adv_suffix):
        return "--unk_adv--"

    return "--unk--"


############################################

# load in the training corpus
with open("/users/shizhengyan/Desktop/Neu/CS6120/hw3/train.pos", 'r') as f:
    training_corpus = f.readlines()
    #print(training_corpus)

# read the vocabulary data, split by each line of text, and save the list
with open("/users/shizhengyan/Desktop/Neu/CS6120/hw3/hmv.txt", 'r') as f:
    voc_l = f.read().split('\n')
    #print(voc_l)

# vocab: dictionary that has the index of the corresponding words
vocab = {} 

# Get the index of the corresponding words. 
for i, word in enumerate(sorted(voc_l)): 
    vocab[word] = i       
    

cnt = 0
for k,v in vocab.items():
    cnt += 1
    if cnt > 20:
        break

# load in the test corpus
with open("/users/shizhengyan/Desktop/Neu/CS6120/hw3/test.pos", 'r') as f:
    y = f.readlines()
    #print(y)

#corpus without tags, preprocessed
_, prep = preprocess(vocab, "/users/shizhengyan/Desktop/Neu/CS6120/hw3/test.words")



###########################################################
'''
task 1 
'''


def create_dictionaries(training_corpus, vocab):
    """
    Params: 
        training_corpus: a corpus where each line has a word followed by its tag.
        vocab: a dictionary where keys are words in vocabulary and value is an index
    Return: 
        emission_counts: a dictionary where the keys are (tag, word) and the values are the counts
        transition_counts: a dictionary where the keys are (prev_tag, tag) and the values are the counts
        tag_counts: a dictionary where the keys are the tags and the values are the counts
    """
    
    # initialize the dictionaries 
    emission_counts=dict()
    
    for i in training_corpus:
        word,tag=get_word_tag(i, vocab)
        #if tag=='``':
            #continue
        if (tag,word) not in emission_counts:
            emission_counts[(tag,word)]=1
        else:
            emission_counts[(tag,word)]+=1
            
            
    transition_counts=dict()
    
    for i in range(len(training_corpus)-1):
        this_tag=get_word_tag(training_corpus[i], vocab)[1]
        next_tag=get_word_tag(training_corpus[i+1], vocab)[1]
        #if this_tag=='``' or next_tag=='``':
            #continue
        if (this_tag,next_tag) not in transition_counts:
            transition_counts[(this_tag,next_tag)]=1
        else:
            transition_counts[(this_tag,next_tag)]+=1
            
        
    tag_counts=dict()
    
    for i in training_corpus:
        word,tag=get_word_tag(i, vocab)
        #if tag=='``':
             #continue
        #if tag=='#' or tag=='$':
            #continue
        if tag not in tag_counts:
            tag_counts[tag]=1
        else:
            tag_counts[tag]+=1
        
        

    # Initialize "prev_tag" (previous tag) with the start state, denoted by '--s--'

        # get the word and tag using the get_word_tag helper function
        
    return emission_counts, transition_counts, tag_counts





emission_counts, transition_counts, tag_counts = create_dictionaries(training_corpus, vocab)
states = sorted(tag_counts.keys())
print(f"Number of POS tags (number of 'states'): {len(states)}")
print("View these POS tags (states)")
print(states)



##########################################################

'''task2
'''

def predict_pos(prep, y, emission_counts, vocab, states):
    '''
    Params: 
        prep: a preprocessed version of 'y'. A list with the 'word' component of the tuples.
        y: a corpus composed of a list of tuples where each tuple consists of (word, POS)
        emission_counts: a dictionary where the keys are (tag,word) tuples and the value is the count
        vocab: a dictionary where keys are words in vocabulary and value is an index
        states: a sorted list of all possible tags for this assignment
    Return: 
        accuracy: Number of times you classified a word correctly
    '''

    num=len(y)
  
    y_tup=[]
    for line in y:
        if not line.split():
            continue
        else:
            word, pos = line.split()
            if word not in vocab: 
                continue
        word,pos=get_word_tag(line,vocab)
        y_tup.append([word,pos])

    L=[]
    for i in range(num):
        max_count=0
        max_tag=None
        for key in list(emission_counts.keys()):
            if key[1]==prep[i]:
                if emission_counts[key] > max_count:
                    max_count=emission_counts[key]
                    max_tag=key[0]
        if [prep[i],max_tag] not in L:
            L.append([prep[i],max_tag])
    
    L=sorted(L,key=lambda x: x[0])
    #print(L)
    y_tup=sorted(y_tup,key=lambda x: x[0])
    correct=0
    total=0
    for i in range(len(L)):
        if L[i][1] not in vocab:
            continue
        for j in range(len(y_tup)):
            if L[i][0]==y_tup[j][0]:
                if y_tup[j][1]==L[i][1]:
                    correct+=1
                total+=1
    accuracy=correct/total
    return accuracy


accuracy_predict_pos = predict_pos(prep, y, emission_counts, vocab, states)
print(f"Accuracy of prediction using predict_pos is {accuracy_predict_pos:.4f}")


# 0.9795

#################################################################
'''task3
'''

def create_transition_matrix(alpha, tag_counts, transition_counts):
    ''' 
    Params: 
        alpha: number used for smoothing
        tag_counts: a dictionary mapping each tag to its respective count
        transition_counts: transition count for the previous word and tag
    Return:
        A: matrix of dimension (num_tags,num_tags)
    '''
    
    states = sorted(tag_counts.keys())
    
    matrix=[[0 for i in range(len(states))] for j in range(len(states))]
    for i in range(len(states)):
        for j in range(len(states)):
            pre_tag=states[i]
            next_tag=states[j]
            if (pre_tag,next_tag) in transition_counts:
                matrix[i][j]=(transition_counts[(pre_tag,next_tag)]+alpha)/(tag_counts[pre_tag]+alpha*len(states))
            else:
                matrix[i][j]=1/len(states)
    return np.matrix(matrix)


alpha = 0.001
A = create_transition_matrix(alpha, tag_counts, transition_counts)
print(A)
# Testing your function
print(f"A at row 0, col 0: {A[0,0]:.9f}")
print(f"A at row 3, col 1: {A[3,1]:.4f}")

print("View a subset of transition matrix A")
A_sub = pd.DataFrame(A[30:35,30:35], index=states[30:35], columns = states[30:35] )
print(A_sub)

C=A.reshape(-1,1)
print('max of transition_matrix: ',max(C))




def create_emission_matrix(alpha, tag_counts, emission_counts, vocab):
    '''
    Params: 
        alpha: tuning parameter used in smoothing 
        tag_counts: a dictionary mapping each tag to its respective count
        emission_counts: a dictionary where the keys are (tag, word) and the values are the counts
        vocab: a dictionary where keys are words in vocabulary and value is an index
    Return:
        B: a matrix of dimension (num_tags, len(vocab))
    '''

    #tag_counts=[i for i in tag_counts.items()]
    states = sorted(tag_counts.keys())
    
    vocab_list=list(vocab.items())

    num_tags=len(tag_counts)
    matrix=[[0 for i in range(len(vocab)+8)] for j in range(num_tags)]
    for i in range(num_tags):
        #tag=tag_counts[i][0]
        tag=states[i]
        #c_tag=tag_counts[i][1] 
        c_tag=tag_counts[states[i]]
        
        for j in range(len(vocab_list)):
            if (tag,vocab_list[j][0]) in emission_counts: 
                c_tag_word=emission_counts[(tag,vocab_list[j][0])]
                matrix[i][j]=(c_tag_word+alpha)/(c_tag+alpha*len(vocab))
            else:
                matrix[i][j]=1/len(vocab)
        if (tag,"--unk_digit--") in emission_counts:
            c_tag_word=emission_counts[(tag,"--unk_digit--")]
            matrix[i][len(vocab_list)]=(c_tag_word+alpha)/(c_tag+alpha*len(vocab))
        else:
            matrix[i][len(vocab_list)]=1/len(vocab)
        if (tag,"--unk_punct--") in emission_counts:
            c_tag_word=emission_counts[(tag,"--unk_punct--")]
            matrix[i][len(vocab_list)+1]=(c_tag_word+alpha)/(c_tag+alpha*len(vocab))
        else:
            matrix[i][len(vocab_list)+1]=1/len(vocab)
        if (tag,"--unk_upper--") in emission_counts:
            c_tag_word=emission_counts[(tag,"--unk_upper--")]
            matrix[i][len(vocab_list)+2]=(c_tag_word+alpha)/(c_tag+alpha*len(vocab))
        else:
            matrix[i][len(vocab_list)+2]=1/len(vocab)
        if (tag,"--unk_noun--") in emission_counts:
            c_tag_word=emission_counts[(tag,"--unk_noun--")]
            matrix[i][len(vocab_list)+3]=(c_tag_word+alpha)/(c_tag+alpha*len(vocab))
        else:
            matrix[i][len(vocab_list)+3]=1/len(vocab)
        if (tag,"--unk_verb--") in emission_counts:
            c_tag_word=emission_counts[(tag,"--unk_verb--")]
            matrix[i][len(vocab_list)+4]=(c_tag_word+alpha)/(c_tag+alpha*len(vocab))
        else:
            matrix[i][len(vocab_list)+4]=1/len(vocab)
        if (tag,"--unk_adj--") in emission_counts:
            c_tag_word=emission_counts[(tag,"--unk_adj--")]
            matrix[i][len(vocab_list)+5]=(c_tag_word+alpha)/(c_tag+alpha*len(vocab))
        else:
            matrix[i][len(vocab_list)+5]=1/len(vocab)
        if (tag,"--unk_adv--") in emission_counts:
            c_tag_word=emission_counts[(tag,"--unk_adv--")]
            matrix[i][len(vocab_list)+6]=(c_tag_word+alpha)/(c_tag+alpha*len(vocab))
        else: 
            matrix[i][len(vocab_list)+6]=1/len(vocab)
        if (tag,"--unk--") in emission_counts:
            c_tag_word=emission_counts[(tag,"--unk--")]
            matrix[i][len(vocab_list)+7]=(c_tag_word+alpha)/(c_tag+alpha*len(vocab))
        else:
            matrix[i][len(vocab_list)+7]=1/len(vocab)

    return np.matrix(matrix)




B = create_emission_matrix(alpha, tag_counts, emission_counts, vocab)
print(f"View Matrix position at row 0, column 0: {B[0,0]:.9f}")
print(f"View Matrix position at row 3, column 1: {B[3,1]:.9f}")
cidx  = ['725','adroitly','engineers', 'promoted', 'synergy']
cols = [vocab[a] for a in cidx]
rvals =['CD','NN','NNS', 'VB','RB','RP']
rows = [states.index(a) for a in rvals]
B_sub = pd.DataFrame(B[np.ix_(rows,cols)], index=rvals, columns = cidx )
print(B_sub)
C=B.reshape(-1,1)
print('max of emission_matrix: ',max(C))


################################################################

'''task 4
'''

# UNQ_C5 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: initialize
def initialize(states, tag_counts, A, B, corpus, vocab):
    '''
    Params: 
        states: a list of all possible parts-of-speech
        tag_counts: a dictionary mapping each tag to its respective count
        A: Transition Matrix of dimension (num_tags, num_tags)
        B: Emission Matrix of dimension (num_tags, len(vocab))
        corpus: a sequence of words whose POS is to be identified in a list 
        vocab: a dictionary where keys are words in vocabulary and value is an index
    Return:
        best_probs: matrix of dimension (num_tags, len(corpus)) of floats
        best_paths: matrix of dimension (num_tags, len(corpus)) of integers
    '''
    # Write your code here 
    num_tags=len(tag_counts)
    best_probs=np.matrix([[0.00 for i in range(len(corpus))] for j in range(num_tags)])
    best_paths=np.matrix([[0.00 for i in range(len(corpus))] for j in range(num_tags)])
    return best_probs, best_paths


best_probs, best_paths = initialize(states, tag_counts, A, B, prep, vocab)
print('shape of best_probs: ', best_probs.shape)
print('shape of best_paths: ', best_paths.shape)
print(f"best_probs[0,0]: {best_probs[0,0]:.4f}") 
print(f"best_paths[2,3]: {best_paths[2,3]:.4f}")



########################################



# TASK CELL

def viterbi_forward(A, B, test_corpus, best_probs, best_paths, vocab):
    '''
    Input: 
        A, B: The transiton and emission matrices respectively
        test_corpus: a list containing a preprocessed corpus
        best_probs: an initilized matrix of dimension (num_tags, len(corpus))
        best_paths: an initilized matrix of dimension (num_tags, len(corpus))
        vocab: a dictionary where keys are words in vocabulary and value is an index 
    Output: 
        best_probs: a completed matrix of dimension (num_tags, len(corpus))
        best_paths: a completed matrix of dimension (num_tags, len(corpus))
    '''

    num_tags = best_probs.shape[0]
    
    vocab_list=list(vocab.items())
    
    for i in range(len(states)):
        if states[i]=="--s--":
            pos=i
    p=vocab[test_corpus[0]]
    #print('p: ',p)
    for x in range(num_tags):
        if A[pos,x]==0 or B[x,p]==0:
            best_probs[x,0]=float('-inf')
        else:
            best_probs[x,0]=np.log(A[pos,x])+np.log(B[x,p]) 
        #best_probs[x,0]=1.000
        #print('A[pos,x]: ',A[pos,x],'B[x,p]: ',B[x,p])
    #for x in range(num_tags):
        #print(best_probs[x,0])
    
    for i in range(1, len(test_corpus)): 
        if i % 5000 == 0:
            print("Words processed: {:>8}".format(i))
        for c in range(num_tags):
            max_value=float('-inf')
            max_pre_tag=-1
            pos=vocab[test_corpus[i]]
            if test_corpus[i] not in vocab:
                pos=assign_unk(test_corpus[i]) 
            for a in range(num_tags):
                #print('A[a,c]: ',A[a,c])
                if best_probs[a,i-1]+np.log(A[a,c])>max_value:
                    max_value=best_probs[a,i-1]+np.log(A[a,c])
                    #print('max_value: ',max_value)
                    max_pre_tag=a
                    #print('B[c,pos]: ',B[c,pos])
                    #print('max_value*B[c,pos]: ',float(max_value*B[c,pos]))
            best_probs[c,i]=max_value+np.log(B[c,pos])
            #print('best_probs[c,i]: ',best_probs[c,i])
            best_paths[c,i]=max_pre_tag
            #print(float(max_value*B[c,pos]))
        
    return best_probs, best_paths


# this will take a few minutes to run => processes ~ 30,000 words
best_probs, best_paths = viterbi_forward(A, B, prep, best_probs, best_paths, vocab)

'''
vocab_list=list(vocab.items())
for i in range(len(vocab_list)):
    if vocab_list[i][0]==prep[0]:
        print('method1: ',i)
print('method2: ',vocab[prep[0]])
'''

# Do not change anything in this cell
print(f"best_probs[0,1]: {best_probs[0,1]:.4f}") 
print(f"best_probs[0,4]: {best_probs[0,4]:.4f}")
print(f"best_probs[10,400]: {best_probs[10,400]:.4f}")



# TASK CELL
def viterbi_backward(best_probs, best_paths, corpus, states):
    '''
    This function returns the best path.
    
    '''
    final_tag_index=list(best_probs[:,-1]).index(max(best_probs[:,-1]))
    trace=[final_tag_index]
    t=final_tag_index
    for i in range(len(prep)-1,0,-1): 
        tag_index=int(best_paths[t,i])
        trace.append(tag_index)
        t=tag_index
    trace=trace[::-1]
    
    #tag_counts=[i for i in tag_counts.items()]
    
    pred=[]
    for index in trace:
        tag=states[index]
        pred.append(tag)
    return pred



pred = viterbi_backward(best_probs, best_paths, prep, states)
m=len(pred)
print('The prediction for pred[-7:m-1] is: \n', prep[-7:m-1], "\n", pred[-7:m-1], "\n")
print('The prediction for pred[0:8] is: \n', pred[0:7], "\n", prep[0:7])

print('The third word is:', prep[3])
print('Your prediction is:', pred[3])
print('Your corresponding label y is: ', y[3])

print('len of pred',m)
print('len of prep',len(prep))

# TASK CELL

def compute_accuracy(pred, y):
    '''
    Params: 
        pred: a list of the predicted parts-of-speech 
        y: a list of lines where each word is separated by a '\t' (i.e. word \t tag)
    Return: 
        accuracy
        
    '''
    correct=0
    total=0
    for i in range(len(y)):
        if not y[i].split():
            word = "--n--"
            tag = "--s--"
        else:
            word, tag = y[i].split()
            total+=1
        if pred[i]==tag:
            correct+=1
        
    accuracy=correct/total
        
    return accuracy


print(f"Accuracy of the Viterbi algorithm is {compute_accuracy(pred, y):.4f}")






