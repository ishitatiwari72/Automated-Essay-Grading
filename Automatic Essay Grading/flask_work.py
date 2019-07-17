# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:28:06 2019

@author: HP
"""

import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('words')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.corpus import words
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn import utils
from sklearn import metrics
from spellchecker import SpellChecker
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from nltk.stem.porter import PorterStemmer
from spellchecker import SpellChecker
from sklearn.decomposition import TruncatedSVD
import string
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re, collections
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import ensemble
import pickle

def clean_text(essay):
    essay=str(essay)
    result = re.sub(r'http[^\s]*', '',essay)
    result = re.sub('[0-9]+','', result).lower()
    result = re.sub('@[a-z0-9]+', '', result)
    return re.sub('[%s]*' % string.punctuation, '',result)



def deEmojify(essay):
    return essay.encode('ascii', 'ignore').decode('ascii')



def get_wordlist(sentence):    
    clean_sentence = re.sub("[^A-Z0-9a-z]"," ", sentence)
    wordlist = nltk.word_tokenize(clean_sentence)    
    return wordlist


def tokenize(essay):
    stripped_essay = essay.strip()
    
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(stripped_essay)    
    tokenized_sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            tokenized_sentences.append(get_wordlist(raw_sentence))    
    return tokenized_sentences


def avg_word_len(essay):    
    clean_essay = re.sub(r'\W', ' ', essay)
    words = nltk.word_tokenize(clean_essay)    
    return sum(len(word) for word in words) / len(words)



# calculating number of words in an essay
def word_count(essay):    
    clean_essay = re.sub(r'\W', ' ', essay)
    words = nltk.word_tokenize(clean_essay)    
    return len(words)



def char_count(essay):    
    clean_essay = re.sub(r'\s', '', str(essay).lower())    
    return len(clean_essay)



def sent_count(essay):    
    sentences = nltk.sent_tokenize(essay)    
    return len(sentences)



def count_lemmas(essay):    
    tokenized_sentences = tokenize(essay)          
    lemmas = []
    wordnet_lemmatizer = WordNetLemmatizer()    
    for sentence in tokenized_sentences:
        tagged_tokens = nltk.pos_tag(sentence)         
        for token_tuple in tagged_tokens:        
            pos_tag = token_tuple[1]
        
            if pos_tag.startswith('N'): 
                pos = wordnet.NOUN
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('J'):
                pos = wordnet.ADJ
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('V'):
                pos = wordnet.VERB
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('R'):
                pos = wordnet.ADV
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            else:
                pos = wordnet.NOUN
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
    
    lemma_count = len(set(lemmas))
    
    return lemma_count




def spell_count(essay):
    spell = SpellChecker()
    essay=essay.split()
    misspelled = spell.unknown(essay)
    return len(misspelled)


def count_pos(essay):
    
    tokenized_sentences = tokenize(essay)
    
    noun_count = 0
    adj_count = 0
    verb_count = 0
    adv_count = 0
    
    for sentence in tokenized_sentences:
        tagged_tokens = nltk.pos_tag(sentence)
        
        for token_tuple in tagged_tokens:
            pos_tag = token_tuple[1]
        
            if pos_tag.startswith('N'): 
                noun_count += 1
            elif pos_tag.startswith('J'):
                adj_count += 1
            elif pos_tag.startswith('V'):
                verb_count += 1
            elif pos_tag.startswith('R'):
                adv_count += 1
            
    return noun_count, adj_count, verb_count, adv_count

def extract_features(data):
    
    features_ = data.copy()
  
    features_['sent_count'] = features_['essay'].apply(sent_count)
  
    features_['essay']=features_['essay'].apply(clean_text)
  
   
    features_['essay']=features_['essay'].apply(deEmojify)
    #features=features.apply(remove_null)
        
    features_['char_count'] = features_['essay'].apply(char_count)
    
    features_['word_count'] = features_['essay'].apply(word_count)
    
   # features_['sent_count'] = features_['essay'].apply(sent_count)
    
    features_['avg_word_len'] = features_['essay'].apply(avg_word_len)
    
    features_['lemma_count'] = features_['essay'].apply(count_lemmas)
    
   # features['spell_err_count'] = features['essay'].apply(count_spell_error)    
    features_['noun_count'], features_['adj_count'], features_['verb_count'], features_['adv_count'] = zip(*features_['essay'].map(count_pos))
    
    features_['spell_count'] = features_['essay'].apply(spell_count)
    return features_

def grade(grade_):
    grade_=int(grade_[0])
    if grade_ in range(0,10):
        return "Your grade is D"
    elif grade_ in range(10,20):
        return "Your grade is C"
    elif grade_ in range(20,50):
        return "Your grade is B"
    elif grade_ in range(50,75):
        return "Your grade is A" 
    else:
        return "your grade is E"

#essay=train_data['essay'][1000]

def Flask_srk(essay):
    data_trial = pd.DataFrame([essay],columns=['essay'])
    features_set1 = extract_features(data_trial)  
    file = open('hardwork.pkl', 'rb')
    cv = pickle.load(file)#feature extraction
    feature1 = cv.transform([essay]).toarray() 
    feature2=features_set1.iloc[:,[2,3,4,5,6,7,8,9]].values  
    feature3 = np.append(feature1, feature2, axis=1)#noun,pronoun,missspell,verb etc
    file1=open("hardwork4.pkl","rb")
    sc1 = pickle.load(file1)         #run from here
    feature_test = sc1.transform(feature3)
    file2 = open("hardwork5.pkl","rb")
    clf2 = pickle.load(file2)   
    prediction1 = clf2.predict(feature_test)#predicting grade
    grade_svr_lsa = grade(prediction1)
    
    
    file3 = open('hardwork1.pkl', 'rb')
    regressor = pickle.load(file3)
    feature4 = cv.transform([essay]).toarray() 
    prediction2 = regressor.predict(feature4) 
    grade_lsa = grade(prediction2)
    
    a=list(features_set1['char_count'])
    b=list(features_set1['sent_count'])
    c=list(features_set1['word_count'])
    d=list(features_set1["lemma_count"])
    e=list(features_set1['spell_count'])
    f=list(features_set1['noun_count'])
    g=list(features_set1['adj_count'])
    h=list(features_set1['verb_count'])
    i=list(features_set1['adv_count'])
    """
    file4 = open('hardwork2.pkl', 'rb')
    sc = pickle.load(file4)
    feature_test_1=features_set1.iloc[:,[2,3,4,5,6,7,8,9]]  
    feature5=sc.transform(feature_test_1)
    file5 = open('hardwork3.pkl', 'rb')
    clf1 = pickle.load(file5)
    prediction3 = clf1.predict(feature5)
    grade_svr = grade(prediction3)
    """
    return(grade_svr_lsa,grade_lsa,a[0],b[0],c[0],d[0],e[0],f[0],g[0],h[0],i[0])
    
    
 
    
    
    