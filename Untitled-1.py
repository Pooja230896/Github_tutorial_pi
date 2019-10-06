# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython

#%% [markdown]
# ### Loading the Data

#%%
import nltk
nltk.download('stopwords') # run this one time
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from string import punctuation
from collections import Counter
import operator
import pandas as pd
from textblob import TextBlob
import numpy as np
import re
import spacy
import gensim
from gensim import corpora, models
pd.set_option("display.max_colwidth", 200)


import operator
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk.stem import WordNetLemmatizer
from string import punctuation
from nltk.tokenize import word_tokenize
from collections import Counter
import operator

# libraries for visualization
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from gensim.models.ldamodel import LdaModel
from pprint import pprint

#import pyldavis
Text = open('../data/C50train/JoeOrtiz/242939newsML.txt').read()
Text

#%% [markdown]
# <img src="../images/icon/ppt-icons.png" alt="Concept-Alert" style="width: 100px;float:left; margin-right:15px"/>
# <br /> 
# 
# ##  Mini-Challenge - 1 
# ***
# ### Instructions
# * Perform a sentence tokenization on the above data using `sent_tokenize()` and store it in a variable called '**Sent**'

#%%
Sent = sent_tokenize(Text)
Sent

#%% [markdown]
# <img src="../images/icon/ppt-icons.png" alt="Concept-Alert" style="width: 100px;float:left; margin-right:15px"/>
# <br /> 
# 
# ##  Mini-Challenge - 2 
# ***
# **Bag of Words** <br>
# In this task, we will try to do the basic NLP operations of tokenizing, removing stop words and lemmatizing on our data. We will also try to create a list of most frequent words
# ### Instructions
# - Iterate over every Sentence in the list **Sent**  using a for loop and convert every sentence into 
#     - lower case 
#     - and then tokenize it using the instantiated object 
# - Now remove the stopwords from the tokens 
# - Lemmatize them using `WordNetLemmatizer()` and save it in `lemmatized_tokens`
# - Append `lemmatized_tokens` into the list called **Text**
# - Convert `Counter(lemmatized_tokens)` into dictionary and save it in a variable called `BoW_dict`.
# - Sort `BoW_dict` in descending order using `sorted()` function with the parameters `BoW_dict.items()`, `key=operator.itemgetter(1)`, `reverse=True`. Store it in a variable called `sorted_d`
# - Finally append them into the list called **Texts** 
# - Print `Texts` to check out the list of words with their frequency in descending order.
# - Print Top 10 words from the `Texts`

#%%
stop = list(stopwords.words('english'))+list(punctuation)
lemma=  WordNetLemmatizer()
Texts = []
Text = []
for i in Sent:
    r = i.lower()
    tokens = word_tokenize(r)
    s = [i for i in tokens if i not in stop]
    lemmatized_tokens = [lemma.lemmatize(i) for i in s]
    Text.append(lemmatized_tokens)
    BoW_dict = dict(Counter(lemmatized_tokens))
    sorted_d = sorted(BoW_dict.items(), key=operator.itemgetter(1), reverse=True)
    Texts.append(sorted_d)
#print(Texts)
print(sorted_d[:10])

#%% [markdown]
# <img src="../images/icon/ppt-icons.png" alt="Concept-Alert" style="width: 100px;float:left; margin-right:15px"/>
# <br /> 
# 
# ##  Mini-Challenge - 3
# ***
# Since Nouns are important in Topic Modeling process, we will try to figure out the top nouns from the bag of words we created in the last task.
# ### Instructions
# - Join the previously created bag of words `lemmatized_tokens` back into a string using `join()` method and store the result in `BoW_joined`.
# - Convert `Bow_joined` into a textblob using `TextBlob()` method and store the result into `blob`.
# - Print out the `blob.tags` to look at the different tags associated with the words.
# - Get the tags of all the words from `lemmatized_tokens` using `blob.tags` and store the result in a variable called `tags`
# - From `tags`, extract the words which have `NN` tags and store them to a list called `nouns`
# - The top 10 words which have appeared most frequently are already stored into a list called `top_words`
# - Compare the two lists `top_words` and `nouns` and store the common elements between them in a new list called `top_nouns`
# - Print `top_nouns` to see most commonly appearing nouns

#%%
nltk.download('averaged_perceptron_tagger')


#%%
top_10 = sorted_d[:10]

#Storing only the top 10 words
top_words=[]
for x in top_10:
    top_words.append(x[0])
    
BoW_joined = ' '.join(lemmatized_tokens)
blob = TextBlob(BoW_joined)
#print(blob.tags)
tags = blob.tags
nouns = []
for x in tags:
    if x[1]=='NN':
        nouns.append(x[0])
    
top_nouns = [x for x in nouns if x in top_words ]
top_nouns = list(set(top_nouns))

print(top_nouns)


#%%
Text

#%% [markdown]
# <img src="../images/icon/ppt-icons.png" alt="Concept-Alert" style="width: 100px;float:left; margin-right:15px"/>
# <br /> 
# 
# ##  Mini-Challenge - 4
# ***
# Using the method `.Dictionary()` inside the module `corpora` to create a unique token for every word and also print out the tokens assigned respectively using the `.token2id` attribute

#%%
dictionary = corpora.Dictionary(Text)
print(dictionary.token2id)

#%% [markdown]
# <img src="../images/icon/ppt-icons.png" alt="Concept-Alert" style="width: 100px;float:left; margin-right:15px"/>
# <br /> 
# 
# ##  Mini-Challenge - 5
# ***
# Now convert the dictionary into a bag of words list using the `.doc2bow()` method in `dictionary` and store it in a variable **corpus** 

#%%
corpus = [dictionary.doc2bow(text) for text in Text]

#%% [markdown]
# <img src="../images/icon/ppt-icons.png" alt="Concept-Alert" style="width: 100px;float:left; margin-right:15px"/>
# <br /> 
# 
# ##  Mini-Challenge - 6
# ***
# Create an LDA model with number of topics as 5 of your choice and your choice of total passes. Now print out the top 5 topics and also the top 3 words in every topic

#%%
#doc_term_matrix = doc2bow(dictionary)

ldamodel = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=50, random_state=20)
pprint(ldamodel.print_topics())


#%%
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
vis

#%% [markdown]
# <img src="../images/icon/quiz.png" alt="Concept-Alert" style="width: 100px;float:left; margin-right:15px"/>
# <br /> 
# 
# ## Topic Modelling
# ***
# 
# Q1. What percentage of the total statements are correct with regards to Topic Modeling?
# ```python
# 1. It is a supervised learning technique
# 2. LDA (Linear Discriminant Analysis) can be used to perform topic modeling
# 3. Selection of number of topics in a model does not depend on the size of data
# 4. Number of topic terms are directly proportional to size of the data
# A) 0
# B) 25
# C) 50
# D) 75
# E) 100
# ans : A
# 
# 
# LDA is unsupervised learning model, LDA is latent Dirichlet allocation, not Linear discriminant analysis. Selection of the number of topics is directly proportional to the size of the data, while number of topic terms is not directly proportional to the size of the data. Hence none of the statements are correct.
# ```
# 
# Q2. In Latent Dirichlet Allocation model for text classification purposes, what does alpha and beta hyperparameter represent-
# ```python
# A) Alpha: number of topics within documents, beta: number of terms within topics False
# B) Alpha: density of terms generated within topics, beta: density of topics generated within terms False
# C) Alpha: number of topics within documents, beta: number of terms within topics False
# D) Alpha: density of topics generated within documents, beta: density of terms generated within topics True
# ans: D
# 
# ```
# Q3. Social Media platforms are the most intuitive form of text data. You are given a corpus of complete social media data of tweets. How can you create a model that suggests the hashtags?
# ```python
# A) Perform Topic Models to obtain most significant words of the corpus
# B) Train a Bag of Ngrams model to capture top n-grams – words and their combinations
# C) Train a word2vector model to learn repeating contexts in the sentences
# D) All of these
# ans :D
# ```
# 

#%%



