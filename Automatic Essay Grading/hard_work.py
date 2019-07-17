# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 18:09:26 2019

@author: HP
"""

#cleaning essay
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
    
    features_['sent_count'] = features_['essay'].apply(sent_count)
    
    features_['avg_word_len'] = features_['essay'].apply(avg_word_len)
    
    features_['lemma_count'] = features_['essay'].apply(count_lemmas)
    
    features_['noun_count'], features_['adj_count'], features_['verb_count'], features_['adv_count'] = zip(*features_['essay'].map(count_pos))
    
#    features_['spell_count'] = features_['essay'].apply(spell_count)
    return features_


####################TRAINING DATA#############################################

train_data=pd.read_csv('training_set_rel3.tsv', delimiter='\t',encoding='ISO-8859-1')
train_data=train_data[['essay_id','essay_set','essay','rater1_domain1','rater2_domain1','rater1_domain2','rater2_domain2','domain1_score','domain2_score']]
train_essay_list1 = train_data['essay'].values
train_data1 = pd.read_csv('training_set_rel3.tsv' ,delimiter="\t",error_bad_lines=False,encoding = 'latin-1')
train_data1 = train_data1.loc[:,['essay_id', 'essay_set', 'essay', 'rater1_domain1', 'rater2_domain1',
       'domain1_score']]

data=train_data.loc[:,['essay', 'rater1_domain1','domain1_score']]
data=data.dropna(axis=0)
#Data Preprocessing
features_set1 = extract_features(data)




#score management from domain1[set 1,3,4,5,6,7,8] and domain2[set 1,2,3,4,5,6,7,8]
   
train_data['domain2_score'] = np.where(np.isnan(train_data['domain2_score']), 0, train_data['domain2_score'])
for i in range(0,12976):
        train_data['score'] = train_data['domain1_score'] + train_data['domain2_score']



##*****************************MODEL_FITTING***********************************##

# Fitting Logistic Regression to the Training set
#Creating the Bag of Words model
# Text to Features 
#lab_enc = preprocessing.LabelEncoder() ->>for more accuracy
#label = np.array(lab_enc.fit_transform(train_data['std_score']))


###################################BAG OF WORDS##################################

#BAG OF WORDS
import pickle 
from sklearn.metrics import cohen_kappa_score

def BOW(essay):
    cv = TfidfVectorizer(max_features = 1500, ngram_range=(1, 3),stop_words='english')  #max feature should be given a limit or not/////
    train_vectors = cv.fit_transform(essay).toarray() 
    feature_names =cv.get_feature_names()
    
    with open("hardwork.pkl","wb") as fp:
        pickle.dump(cv,fp)         

    return(feature_names,train_vectors)


feature_name_train,train_vectors=BOW(data['essay'])



###############################DICTIONARY###########################
dataframe = pd.DataFrame.from_dict({w: features[:, i] for i, w in enumerate(feature_name)})


##############SPLITTING DATA########################
# label=np.array(train_set[0]['score'],dtype=int)-->>for 1 set
label=train_data['score'].as_matrix()
featuretrain,featuretest,labeltrain,labeltest=train_test_split(train_vectors,label,test_size=0.3)


######################3LINEAR REGRESSOR####################
regressor = LinearRegression()  
regressor.fit(featuretrain, labeltrain) 

labels_pred = regressor.predict(featuretest) 
regressor.score(featuretrain,labeltrain)
with open("hardwork1.pkl","wb") as fp1:
    pickle.dump(regressor,fp1)         

"""
############################SVD_LSA##########################

svd = TruncatedSVD(n_components=100, random_state=0)
train_svd=svd.fit(featuretrain)  
svd1 = TruncatedSVD(n_components=100, random_state=0)
test_svd=svd1.fit_transform(featuretest)  

##################################    SVR     #############################
clf = SVR(gamma='scale', C=1.0, epsilon=0.2)
clf.fit(featuretrain,labeltrain) 
label_pred = clf.predict(featuretest)
clf.score(featuretrain,labeltrain)
"""
print('Mean Squared Error:', metrics.mean_squared_error(labeltest, labels_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(labeltest, labels_pred))) 
print("Cohen's kappa score:",  cohen_kappa_score(np.rint(labels_pred), labeltest))



#####################################SVR_MODEL################################3
feature=features_set1.iloc[:,4:]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
feature_train,feature_test,label_train,label_test=train_test_split(feature,label,test_size=0.3)
# Feature Scaling
feature_train = sc.fit_transform(feature_train)
feature_test = sc.transform(feature_test)
with open("hardwork2.pkl","wb") as fp2:
    pickle.dump(sc,fp2)         

clf1 = SVR(gamma='scale', C=1.0, epsilon=0.2)
clf1.fit(feature_train, label_train)
label_pred = clf1.predict(feature_test) 
clf1.score(feature_train,label_train)
with open("hardwork3.pkl","wb") as fp3:
    pickle.dump(clf1,fp3)         



##############################SVR+LSA#########################################
###############################################################################
feature_name_train,train_vectors=BOW(data['essay'])
feature1=feature.values

feature2 = np.append(feature1, train_vectors, axis=1)

label_=train_data['score'].as_matrix()
sc1 = StandardScaler()
feature_train_,feature_test_,label_train_,label_test_=train_test_split(feature2,label_,test_size=0.3)
feature_train_ = sc1.fit_transform(feature_train_)
feature_test_ = sc1.transform(feature_test_)
with open("hardwork4.pkl","wb") as fp4:
    pickle.dump(sc1,fp4)         


clf2 = SVR(gamma='scale', C=1.0, epsilon=0.2)
clf2.fit(feature_train_, label_train_)
label_pred2 = clf2.predict(feature_test_)
with open("hardwork5.pkl","wb") as fp5:
    pickle.dump(clf2,fp5)         
