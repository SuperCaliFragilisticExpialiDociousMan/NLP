# -*- coding: utf-8 -*-

# Do not change anything in this cell
import re
import string
from os import getcwd


import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords, twitter_samples
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms 

import json


# TASK 1 CELL

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
            word=stemmer.stem(word)
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


##############################

# TASK 1 CELL

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


###################################


# TASK 2 CELL

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


print(frequency_dict)



##########################################################


# TASK 3 CELL

def train_naive_bayes(frequency_dict, train_x, train_y):
    '''
    Params: 
        frequency_dict: dictionary from (word, label) to how often the word appears
        train_x: a list of tweets
        train_y: a list of labels correponding to the tweets (0,1)
    Return:
        logprior: the log prior. (equation 3 above)
        loglikelihood: the log likelihood of you Naive bayes equation. (equation 6 above)
    '''
    loglikelihood = dict()
    logprior = 0


    # calculate V, the number of unique words in the vocabulary 
    
    vocab = []
    for i in frequency_dict.keys():
        if i[0] not in vocab:
            vocab.append(i[0])
    V = len(vocab)
    
    #print(vocab) 

    # calculate num_pos and num_neg - the total number of positive and negative words for all documents
    num_pos = num_neg = 0
    for pair in frequency_dict.keys():
        # if the label is positive (greater than zero)
        if pair[1]>0:

            # Increment the number of positive words by the count for this (word, label) pair
            num_pos += frequency_dict[pair]

        # else, the label is negative
        else:

            # increment the number of negative words by the count for this (word,label) pair
            num_neg += frequency_dict[pair] 

    # Calculate num_doc, the number of documents
    num_doc = len(train_y)

    # Calculate D_pos, the number of positive documents 
    pos_num_docs = sum(train_y)

    # Calculate D_neg, the number of negative documents 
    neg_num_docs = num_doc-pos_num_docs 

    # Calculate logprior
    logprior = np.log(pos_num_docs/neg_num_docs)
    
    for word in vocab:
        freq_pos=find(frequency_dict, word, 1)
        freq_neg=find(frequency_dict, word, 0)
        pro_pos=(freq_pos+1)/(V+num_pos)
        pro_neg=(freq_neg+1)/(V+num_neg)
        loglikelihood[word]=np.log(pro_pos/pro_neg)
        

    return logprior, loglikelihood


#############################################################################


# Do not change anything in this cell
logprior, loglikelihood = train_naive_bayes(frequency_dict, train_x, train_y)

print(logprior)
print(loglikelihood)


###################################################################


# TASK 4 CELL

def naive_bayes_predict(tweet, logprior, loglikelihood):
    '''
    Params:
        tweet: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Return:
        total_prob: the sum of all the logliklihoods of each word in the tweet (if found in the dictionary) + logprior (a number)

    '''
    # process the tweet to get a list of words
    word_l = clean_tweet(tweet)

    # initialize probability to zero
    total_prob = 0

    # add the logprior
    total_prob += logprior


    for word in word_l:
        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            total_prob  += loglikelihood[word]



    return total_prob
    
##################################################

    
# Do not change anything in this cell
custom_tweet = 'I love NLP'
prob = naive_bayes_predict(custom_tweet, logprior,  loglikelihood)
print('The expected output is', prob)

###########################################################


# TASK 5 CELL

def test_naive_bayes(test_x, test_y, logprior, loglikelihood):
    """
    Params:
        test_x: A list of tweets
        test_y: the corresponding labels for the list of tweets
        logprior: the logprior
        loglikelihood: a dictionary with the loglikelihoods for each word
    Return:
        accuracy: (# of tweets classified correctly)/(total # of tweets)
    """
    accuracy = 0  

    y_hats = []
    for tweet in test_x:
        # if the prediction is > 0
        if naive_bayes_predict(tweet, logprior, loglikelihood) > 0:
            # the predicted class is 1
            y_hat_i = 1
        else:
            # otherwise the predicted class is 0
            y_hat_i = 0

        # append the predicted class to the list y_hats
        y_hats.append(y_hat_i) 

    # error is the average of the absolute values of the differences between y_hats and test_y
    error = 0
    for i in range(len(test_y)):
        error+=abs(test_y[i]-y_hats[i])
    error/=len(test_y)

    num_of_predict_correct=0
    accuracy = 0
    for i in range(len(test_y)):
        if test_y[i]==y_hats[i]:
            num_of_predict_correct+=1
            
    accuracy = num_of_predict_correct/len(test_y)
    print(y_hats)

    return accuracy 

###############################################################

# Do not change anything in this cell
print("Naive Bayes accuracy = %0.4f" %
      (test_naive_bayes(test_x, test_y, logprior, loglikelihood)))

 





