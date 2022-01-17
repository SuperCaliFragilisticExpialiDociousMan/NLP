#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 18:56:13 2021

@author: shizhengyan
"""

import re
import string
from os import getcwd

import pdb

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords, twitter_samples
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


#####################################################
# Task0.1

def clean_tweet(tweet):
    '''
    Params:
        tweet: a string containing a tweet
    Return:
        tweets_clean: a list of words containing the processed tweet

    ''' 
    stemmer=PorterStemmer()
    tokenizer = TweetTokenizer()
    
    #regular expression
    
    tweet=re.sub(r'#', '', tweet)
 
    
    tweet_tokens = tokenizer.tokenize(tweet) 
    #print(tweet_tokens)
    tweets_cleaned = []
    for word in tweet_tokens:
        word=stemmer.stem(word)
        if word not in stopwords.words('english'):
            #word=stemmer.stem(word)
            tweets_cleaned.append(word)
        

        
    return tweets_cleaned



# Helper function to find word frequency
def find(frequency, word, label):
    '''
    Params:
        frequency: a dictionary with the frequency of each pair (or tuple)
        word: the word to look up
        label: the label corresponding to the word
    Return:
        n: the number of times the word with its corresponding label appears.
    '''
    n = 0  
    
    word_pair = (word,label)
    if word_pair not in frequency:
        n = 0
    else:
        n=frequency[word_pair]
    return n



################################

# Run this cell
nltk.download('stopwords')
nltk.download('twitter_samples')


###############################

# add folder, tmp2, from our local workspace containing pre-downloaded corpora files to nltk's data path
filePath = f"{getcwd()}/../tmp2/"
nltk.data.path.append(filePath)




data1=twitter_samples.strings('/users/shizhengyan/nltk_data/corpora/twitter_samples/positive_tweets.json')
data2=twitter_samples.strings('/users/shizhengyan/nltk_data/corpora/twitter_samples/negative_tweets.json')


# get the sets of positive and negative tweets
positive_tweets = data1
all_negative_tweets = data2




train_list_d=[]
train_list_d.extend(data1[0:len(data1)-50])
train_list_d.extend(data2[0:len(data2)-50])

test_list_d=[]
test_list_d.extend(data1[-50:])
test_list_d.extend(data2[-50:])


train_list_c=[1] * (len(data1)-50)
train_list_c.extend([0] * (len(data2)-50))
test_list_c=[1]*50
test_list_c.extend([0]*50)


# split the data into train and validation set
train_x = train_list_d
test_x = test_list_d

train_y = train_list_c
test_y = test_list_c


# split the data into train and validation set


######################################################
# Task0.2

def tweet_counter(output, tweets, tweet_senti):
    '''
    Params:
        output: a dictionary that will be used to map each pair to its frequency
        tweets: a list of tweets
        tweet_senti: a list corresponding to the sentiment of each tweet (either 0 or 1)
    Return:
        output: a dictionary mapping each pair to its frequency
    '''
 
    for label, tweet in zip(tweet_senti, tweets):
        for word in clean_tweet(tweet):
            if (word,label) not in output:
                output[(word,label)]=1
            else:
                output[(word,label)] += 1
    
    #print(output)
    return output

output=dict()


tweets=train_x
tweet_senti=train_y

frequency_dict = tweet_counter(output, tweets, tweet_senti) 


############################################################
import math

# Task 1.1

def sigmoid(z): 
    '''
    Input:
        z: is the input (can be a scalar or an array)
    Output:
        h: the sigmoid of z
    '''
    
   
    # calculate the sigmoid of z
    h = 1/(1+math.e**(-z))
    
    
    return h

print(sigmoid(0))
print(sigmoid(4.92))

# Task1.2

def gradientDescent(x, y, theta, alpha, num_iters):
    '''
    This function should return (J, theta) by taking (x, y, theta, alpha, num_iters) as inputs, where
        x: matrix of features which is (m,n+1)
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations you want to train your model for
        J: the final cost
        theta: your final weight vector
    '''
    
    # get 'm', the number of rows in matrix x
    m = len(x)
    
    for i in range(0, num_iters):
        
        # get z, the dot product of x and theta
        z = np.dot(x, theta) 
        
        # get the sigmoid of z
        h = sigmoid(z)

        # calculate the cost function 
        
        J = -(np.dot(y.T,np.log(h))+np.dot((1-y).T,np.log(1-h)))/m
        # update the weights theta
        theta= theta-alpha*(np.dot(x.T,(h-y)))/m

        
    ### END CODE HERE ###
    J = float(J)
    return J, theta





# expected output should match with your output
# Check the function
# Construct a synthetic test case using numpy PRNG functions
np.random.seed(1)
# X input is 10 x 3 with ones for the bias terms
tmp_X = np.append(np.ones((10, 1)), np.random.rand(10, 2) * 2000, axis=1)
# Y Labels are 10 x 1
tmp_Y = (np.random.rand(10, 1) > 0.35).astype(float)

print('type(tmp_X)',type(tmp_X))
print('type(tmp_Y)',type(tmp_Y))
print(tmp_X.shape)
print(tmp_Y.shape)
# Apply gradient descent
tmp_J, tmp_theta = gradientDescent(tmp_X, tmp_Y, np.zeros((3, 1)), 1e-8, 700)
print(f"The cost after training is {tmp_J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(tmp_theta)]}")


#######################################################

# Task2

def extract_features(tweet, freqs):
    '''
    This function should take tweet, freqs as input and return x, where:
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        x: a feature vector of dimension (1,3)
    '''
    # process_tweet tokenizes, stems, and removes stopwords
    word_l = clean_tweet(tweet)
    
    
    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros(3).reshape(1,3)
    
    #bias term is set to 1
    x[0,0] = 1 
    
    
    # loop through each word in the list of words
    for word in word_l:
        
        # increment the word count for the positive label 1
        x[0,1] += find(freqs, word, 1)
        
        # increment the word count for the negative label 0
        x[0,2] += find(freqs, word, 0)
        
    assert(x.shape == (1, 3))
    return x


# Check your function
# expected output should match with your output
# test1: on training data
tmp1 = extract_features(train_x[0], frequency_dict)
print(tmp1)

# test 2:
# expected output should match with your output
# check for when the words are not in the freqs dictionary
tmp2 = extract_features('blorb bleeeeb bloooob', frequency_dict)
print(tmp2)



######################################################

#  Task 3 

# collect the features 'x' and stack them into a matrix 'X'
# Expected output should map to your output
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :]= extract_features(train_x[i],frequency_dict)
print(X.shape)
print(X[0])
# training labels corresponding to X
Y = (np.array(train_y)).reshape(len(train_y),1)
print(Y.shape)
print('type(X)',type(X))
print('type(Y)',type(Y))
# Apply gradient descent
J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-8, 700) #call gradient descent by appropriate parameters.
print(f"The cost after training is {J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")




#####################################################

# Task 4

def predict_tweet(tweet, freqs, theta):
    '''
    This function should take (tweet, freqs, theta) as input and return y_pred as output where
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights
        y_pred: the probability of a tweet being positive or negative
    '''
    
    # extract the features of the tweet and store it into x
    x = extract_features(tweet, freqs)    # call extract features function with appropriate parameters 
    
    # make the prediction using x and theta by calling sigmoid function
    z = np.dot(x, theta) 

    y_pred = sigmoid(z)
    
    return y_pred



# Run this cell to test your function
# Expected output should match to your output.
for tweet in ['I am happy', 'I am bad', 'this movie should have been great.', 'great', 'great great', 'great great great', 'great great great great']:
    print( '%s -> %f' % (tweet, predict_tweet(tweet, frequency_dict, theta)))
 
    

my_tweet = ['I am tired', 
            'You are excited', 
            'this movie should have been great.', 
            'funny', 
            'supercalifragilisticexpialidocious', 
            'ridiculous', 
            'sicbudoqocbdsbibcaibnde', 
            'Are you able to handle this?']  # give your custom input and check    

for tweet in my_tweet:
    print( '%s -> %f' % (tweet, predict_tweet(tweet, frequency_dict, theta)))



def test_logistic_regression(test_x, test_y, freqs, theta):
    """
    This function should take (test_x, test_y, freqs, theta) as input and return accuracy as output, where
        test_x: a list of tweets
        test_y: (m, 1) vector with the corresponding labels for the list of tweets
        freqs: a dictionary with the frequency of each pair (or tuple)
        theta: weight vector of dimension (3, 1)
        accuracy: (# of tweets classified correctly) / (total # of tweets)
    """
    
    # the list for storing predictions
    y_hat = []
    
    for tweet in test_x:
        # get the label prediction for the tweet
        y_pred = predict_tweet(tweet, freqs, theta)  # call predict tweet function with proper parameters
        
        if y_pred > 0.5:
            y_hat.append(1.0)
            # append 1.0 to the list
            
        else:
            y_hat.append(0)
            # append 0 to the list
            


    # With the above implementation, y_hat is a list, but test_y is (m,1) array
    # convert both to one-dimensional arrays in order to compare them using the '==' operator
    correct_num=0
    for i in range(len(y_hat)):
        if y_hat[i]==test_y[i]:
            correct_num+=1
    accuracy = correct_num/len(test_y)    # use squeeze function of numpy array on test_y and use appropiate expression.
    
    return accuracy


#Expected output should match to your output. 
tmp_accuracy = test_logistic_regression(test_x, test_y, frequency_dict, theta)
print(f"Logistic regression model's accuracy = {tmp_accuracy:.4f}")




###########################################################
    
# Task 5
    
#Take a tweet and predict whether it is Positive Sentiment ot Negative Sentiment
my_tweet = 'Huge thank you to my new @TheSpringHillCo partners RedBird, @FenwaySports, @Nike, and @EpicGames.  We are on a mission to empower and with this group...Imagine how many lives we can change!!!'  
#string containing the tweet
print(extract_features(my_tweet,frequency_dict))
y_hat = predict_tweet(my_tweet, frequency_dict, theta)  # use appropriate function call to get prediction
print(y_hat)
if y_hat > 0.5:
    print('Positive sentiment')
else: 
    print('Negative sentiment')




