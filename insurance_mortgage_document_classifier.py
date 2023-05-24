#!/usr/bin/env python
# coding: utf-8

# ## Count Vector, TFIDF Representations of Text
# 
# Working with text generally involves converting it into a format that our model is able to understand, which is mostly numbers. In this notebook, you will have a closer look on two of the most basic and ubiquitiously used formats: 
# 
#  - Count Vector
#  - TFIDF
# 
# You will also build a Machine Learning model on a real world dataset of **BBC News** and perform text classification utilizing the above two formats.
# 
# #### Table of Contents
# 1. About the Dataset
# 2. Preprocessing Text
# 3. Working with Count Vector
# 4. Using TFIDF to improve Count Vector
# 5. Conclusion
# 6. Challenge

# ### 1. About the Dataset
# 
# The dataset that you are going to use is a collection of news articles from BBC across 5 major categories, namely:
#  
#  - Business
#  - Entertainment
#  - Politics
#  - Sport
#  - Tech
# 
# There are a total of 2225 articles in the dataset, which is a mix of all of the above categories. Let's load the dataset using pandas and have a quick look at some of the articles. 
# 
# **Note:** You can get the dataset [here](https://trainings.analyticsvidhya.com/asset-v1:AnalyticsVidhya+LP_DL_2019+2019_T1+type@asset+block@bbc_news_mixed.csv)
# 

# In[8]:


import os
from PyPDF2 import PdfReader
import pandas as pd


# In[9]:


def pdf_to_text(pdf_path):
    reader = PdfReader(pdf_path)

    # printing number of pages in pdf file
    # print(len(reader.pages))

    # getting a specific page from the pdf file
    page = reader.pages[0]

    # extracting text from page
    text = page.extract_text()
    # print(text)
    return text


# In[24]:


df = pd.DataFrame(columns=['text', 'label'])

# In[25]:


mortgage_doc_dir = 'mortgage_documents'
label = 'mortgage'
for filename in os.listdir(mortgage_doc_dir):
    f = os.path.join(mortgage_doc_dir, filename)
    # checking if it is a file
    if os.path.isfile(f):
        # print(f)
        pdf_text = pdf_to_text(f)
        df = df.append({'text': pdf_text, 'label': label}, ignore_index=True)

# In[26]:


insurance_doc_dir = 'insurance_documents'
label = 'insurance'
for filename in os.listdir(insurance_doc_dir):
    f = os.path.join(insurance_doc_dir, filename)
    # checking if it is a file
    if os.path.isfile(f):
        # print(f)
        pdf_text = pdf_to_text(f)
        df = df.append({'text': pdf_text, 'label': label}, ignore_index=True)

import pandas as pd

# Load the dataset
bbc_news = pd.read_csv('bbc_news_mixed.csv')
bbc_news.head()

# In[28]:


# print first 2 articles
for art in df.text[:2]:
    print(art)

# Now that you have an idea of how your data looks like, let's see the count of each category in the dataset!

# In[30]:


# category-wise count
df.label.value_counts()

# ### 2. Preprocessing Text
# 
# You would have noticed that the labels are in text format, in order to build a model on this dataset you will have to create a mapping between the labels and numbers like 0,1,2,3 this process is called Label Encoding. You can easily label encode your text data using sklearn's [LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html). Let's have a look at how to do that!

# In[31]:


from sklearn.preprocessing import LabelEncoder

# initialize LabelEncoder
lencod = LabelEncoder()
# fit_transform() converts the text to numbers
df.label = lencod.fit_transform(df.label)
# label-wise count
df.label.value_counts()

# **Note** You'd have noticed in the output of the above code that the text labels have been replaced by numbers. We have a mapping like this -
#  - 0 is Business
#  - 1 is Entertainment
#  - 2 is Politics
#  - 3 is Sport
#  - 4 is Tech
#  
# ### 3. Working with Count Vector
# 
# Sklearn provides an easy way to create count vectors from a piece of text. You can use the [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) to do that. Let's see how simple it is!

# In[32]:


from sklearn.feature_extraction.text import CountVectorizer

# initialize count vector
cvec = CountVectorizer(stop_words='english')
# create Bag of Words
bow = cvec.fit_transform(df.text)
# shape of Bag of Words
print('shape of BOW:', bow.shape)
# number of words in the vocabulary
print('No. of words in vocabulary:', len(cvec.vocabulary_))

# Let's have a closer look at the Bag of Words that you have just generated.

# In[34]:


# create a dataframe from the BOW
# bow_df = pd.SparseDataFrame(bow, columns=cvec.get_feature_names(), index=bbc_news.index, default_fill_value=0)
bow_df = pd.DataFrame.sparse.from_spmatrix(bow, columns=cvec.get_feature_names(), index=df.index)
# sample some data points
bow_df.iloc[:10, :50]

# If you explore the above dataframe, you will find that the Bag of Words representation of the text. Notice that the word "called" appears in the first only once hence there is a 1 at it's index. Now that your BOW is created, let's see just how good is it at classifying the articles in a ML model.
# 
# You'll be using [MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html) model because it works well with sparse features of text.

# In[35]:


df.head()

# In[36]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# creates a ML model based on parameters
def create_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = MultinomialNB()
    model = model.fit(X_train, y_train)
    return model, X_test, y_test


# In[37]:


# create BOW based classification model
model_b, X_test_b, y_test_b = create_model(bow, df.label)

# Now that the model is created and trained, have a look at the classification accuracy:

# In[38]:


from sklearn.metrics import accuracy_score

# check accuracy 
accuracy_score(y_test_b, model_b.predict(X_test_b))

# That's a pretty good accuracy and now let's see how can you improve it even further!
# 
# ### 4. Using TFIDF to improve Count Vector
# 
# Just like Count Vector, TFIDF can also be very easily implemented in Python using Sklearn. Here's how you will create a TFIDF representation of your text.

# In[14]:


from sklearn.feature_extraction.text import TfidfVectorizer

# initialize TFIDF
vec = TfidfVectorizer(max_features=4000, stop_words='english')
# create TFIDF
tfidf = vec.fit_transform(bbc_news.text)
# shape of TFIDF
tfidf.shape

# **Note** this time you have a smaller number of columns in shape(4000 as compared to 29192 of previous). This is because we have used the paremeter `max_features` which tells Sklearn to only use 4000 most important words from the entire text in the dataset to build the TFIDF representation. Have a look at how it looks like!

# In[16]:


# create a dataframe from the TFIDF
# tfidf_df = pd.SparseDataFrame(tfidf, columns=vec.get_feature_names(), index=bbc_news.index, default_fill_value=0)
tfidf_df = pd.DataFrame.sparse.from_spmatrix(tfidf, columns=vec.get_feature_names(), index=bbc_news.index)

# sample some data points
tfidf_df.iloc[:20, 1000:1050]

# If you explore the above dataframe, you will find the TFIDF representation of the text. Notice that each word has a numeric value associated with it, with respect to a column(that in turn represents each document), this is the TFIDF score of that word. Now that your TFIDF is created, let's see just how good is it at classifying the articles in a ML model.

# In[17]:


# create TFIDF based classification model
model_t, X_test_t, y_test_t = create_model(tfidf, bbc_news.label)

# In[18]:


from sklearn.metrics import accuracy_score

# check accuracy
accuracy_score(y_test_t, model_t.predict(X_test_t))

# ### 5. Conclusion
# 
#  - Notice that using TFIDF word presentation, you were able to build a better model by just using 4000 words as oppossed to the 29,192 words of the BOW. 
#  - This is where TFIDF's strength lies which gives the intution that rest of the 25,000+ words weren't adding much useful information to the model and would be common among many documents.
#  - You can know more about the word vectors, TFIDF and similar text embeddings in [this comprehensive article](https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/).
#  - Finally, note that you could have gotten an even better accuracy by doing preprocessing over the text like Normalization, spelling correction and much more.

# ### 6. Challenge
# 
# If you notice the TFIDF dataframe, words like `demand` `demands` and `demanded` are counted separately this is because the data set isn't normalize yet. I encourage you to go ahead and try to do that using concepts learnt in the previous classes.

# In[ ]:


# Your code here
