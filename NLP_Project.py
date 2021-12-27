#!/usr/bin/env python
# coding: utf-8

# In[382]:


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import utils
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
get_ipython().run_line_magic('matplotlib', 'inline')


# # Read in Data

# In[303]:


data = pd.read_json("News_Category_Dataset_v2.json",lines=True)


# ## Remove Stopwords

# In[10]:


def clean_text(text):
    """
    Takes in headline or short description
    Remove punctuation, stopwords
    Return clean text
    """
    
    rmv_punc = [x for x in text if x not in string.punctuation]
    
    rmv_punc = "".join(rmv_punc)
    
    return[word for word in rmv_punc.split() if word.lower() not in stopwords.words("english")]


# # Split into train and test data

# Train will be 2012, 2013, 2015 - 2017, test is 2014, 2018

# In[304]:


data["year"] = data["date"].dt.year


# In[383]:


data["category"].unique()


# # Merge categories together

# In[306]:


data.loc[data["category"] == "MEDIA", "category"] = "ENTERTAINMENT"
data.loc[data["category"] == "WORLDPOST", "category"] = "WORLD NEWS"
data.loc[data["category"] == "THE WORLDPOST", "category"] = "WORLD NEWS"
data.loc[data["category"] == "WEIRD NEWS", "category"] = "MISC"
data.loc[data["category"] == "GOOD NEWS", "category"] = "MISC"
data.loc[data["category"] == "FIFTY", "category"] = "MISC"
data.loc[data["category"] == "RELIGION", "category"] = "MISC"
data.loc[data["category"] == "COLLEGE", "category"] = "EDUCATION"
data.loc[data["category"] == "CULTURE & ARTS", "category"] = "ARTS & CULTURE"


# In[307]:


data.loc[data["category"] == "ARTS", "category"] = "ARTS & CULTURE"
data.loc[data["category"] == "STYLE", "category"] = "STYLE & BEAUTY"
data.loc[data["category"] == "LATINO VOICES", "category"] = "DIFFERENT VOICES"
data.loc[data["category"] == "WOMEN", "category"] = "DIFFERENT VOICES"
data.loc[data["category"] == "BLACK VOICES", "category"] = "DIFFERENT VOICES"
data.loc[data["category"] == "QUEER VOICES", "category"] = "DIFFERENT VOICES"
data.loc[data["category"] == "GREEN", "category"] = "ENVIRONMENT"
data.loc[data["category"] == "MONEY", "category"] = "BUSINESS"
data.loc[data["category"] == "TASTE", "category"] = "FOOD & DRINK"
data.loc[data["category"] == "TECH", "category"] = "SCIENCE & TECH"
data.loc[data["category"] == "SCIENCE", "category"] = "SCIENCE & TECH"
data.loc[data["category"] == "DIVORCE", "category"] = "MARRIED LIFE"
data.loc[data["category"] == "WEDDINGS", "category"] = "MARRIED LIFE"
data.loc[data["category"] == "PARENTS", "category"] = "MARRIED LIFE"
data.loc[data["category"] == "PARENTING", "category"] = "MARRIED LIFE"
data.loc[data["category"] == "HEALTHY LIVING", "category"] = "WELLNESS"


# In[309]:


test = data[(data["year"]==2014) | (data["year"] == 2018)]
train = data[(data["year"]!=2014) & (data["year"] != 2018)]


# In[310]:


train["category"].value_counts()


# In[311]:


test["category"].value_counts()


# In[313]:


data["category"].value_counts()


# # Combine Description and headline into 1 + Clean Text

# In[19]:


train_set = train[["category","headline","short_description"]]
train_set["Text"] = train_set["headline"] + " " + train_set["short_description"]
test_set = test[["category","headline","short_description"]]
test_set["Text"] = test_set["headline"] + " "+ test_set["short_description"]


# In[10]:



train_set["Text"] = train_set["Text"].apply(clean_text)
test_set["Text"] = test_set["Text"].apply(clean_text)


# In[384]:


test_set.to_csv("text_test_set1",index=False)
train_set.to_csv("text_train_set1",index=False)


# # Output to CSV to save time, now just load it in

# In[363]:


text_train_set = pd.read_csv("text_train_set1")
text_test_set =  pd.read_csv("text_test_set1")


# In[370]:


text_test_set = text_test_set[["category", "Text"]]
text_train_set = text_train_set[["category", "Text"]]


# In[374]:


def remove_non_ascii(text):
    return "".join(i for i in text if ord(i)<128)


# In[375]:


text_test_set["Text"] = text_test_set["Text"].apply(remove_non_ascii)
text_train_set["Text"] = text_train_set["Text"].apply(remove_non_ascii)


# In[376]:


text_train_set = shuffle(text_train_set)
text_test_set = shuffle(text_test_set)


# # Create Model

# # Naive Bayes
# 

# In[394]:


X_train_NB, X_val_NB, y_train_NB, y_val_NB = train_test_split(text_train_set['Text'], text_train_set['category'], test_size=0.2)


# In[398]:


params = np.linspace(0,1,11)


# In[402]:


for i in params:
    pipeline = Pipeline([
    ('bow', CountVectorizer(max_features=50000)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB(i)),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
    pipeline.fit(X_train_NB,y_train_NB)

    predictions = pipeline.predict(X_val_NB)
    print(classification_report(predictions,y_val_NB))


# In[403]:


pipeline = Pipeline([
    ('bow', CountVectorizer(max_features=50000)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB(0.1)),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
pipeline.fit(X_train_NB,y_train_NB)

predictions = pipeline.predict(X_val_NB)
print(classification_report(predictions,y_val_NB))


# In[404]:


predictions_test = pipeline.predict(text_test_set["Text"])

print(classification_report(predictions_test,text_test_set["category"]))


# # RNN

# # Encode as Tokenizer

# In[418]:


MaxTokens = 50000
encoder = tf.keras.preprocessing.text.Tokenizer(num_words=MaxTokens,)


# In[419]:


encoder.fit_on_texts(text_train_set["Text"].values)


# In[420]:


word_index = encoder.word_index


# In[421]:


max_size = 0
for i in text_train_set["Text"]:
    if len(i.split()) > max_size:
        max_size = len(i.split())
    else:
        max_size = max_size


# In[422]:


max_size


# In[423]:


max_size1 = 0
for i in text_test_set["Text"]:
    if len(i.split()) > max_size:
        max_size1 = len(i.split())
    else:
        max_size1 = max_size1


# In[415]:


new_max = max(max_size,max_size1)


# In[425]:


new_max


# # Encode to Sequences

# In[424]:


X_train = encoder.texts_to_sequences(text_train_set["Text"].values)
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen = new_max)
X_test = encoder.texts_to_sequences(text_test_set["Text"].values)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen = new_max)


# In[38]:


y_train = pd.get_dummies(text_train_set["category"]).values)


# In[426]:


y_test = pd.get_dummies(text_test_set["category"].values)


# # Models

# # Many models were wokred on and with different params, only the selected best performing are shown below

# In[435]:


tf.random.set_seed(123)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
    MaxTokens, 100, input_length = X_train.shape[1] ),
    tf.keras.layers.SpatialDropout1D(0.2),
    tf.keras.layers.LSTM(100, dropout=0.2,recurrent_dropout = 0.2),
    tf.keras.layers.Dense(20, kernel_regularizer=tf.keras.regularizers.l1_l2(0,0),activation="softmax")

])
model.summary()

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(0.005),
            metrics=["accuracy"])
model.fit(X_train, y_train, epochs=5, batch_size=64,validation_split=0.2,
         callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=1,verbose=0, mode='auto'))


# In[436]:


acc = model.evaluate(X_test,y_test)


# # A way to visualize how predictions work

# In[ ]:


pred = model.predict(X_test)


# In[ ]:


max(pred[0])


# In[162]:


pred[0]


# # Included some other RNN's in the code, but these results aren't in report except for GRU at end

# In[120]:


tf.random.set_seed(123)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
    MaxTokens, 50, input_length = X_train.shape[1] ),
    tf.keras.layers.SpatialDropout1D(0.2),
    tf.keras.layers.LSTM(50, dropout=0.2,recurrent_dropout = 0.2),
    tf.keras.layers.Dense(20, kernel_regularizer=tf.keras.regularizers.l1_l2(0,0),activation="softmax")

])

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(0.01),
            metrics=["accuracy"])
model.fit(X_train, y_train, epochs=5, batch_size=64,validation_split=0.2,
         callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=2,verbose=0, mode='auto'))  


# In[123]:


tf.random.set_seed(123)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
    MaxTokens, 50, input_length = X_train.shape[1] ),
    tf.keras.layers.SpatialDropout1D(0.2),
    tf.keras.layers.LSTM(50, dropout=0.2,recurrent_dropout = 0.2),
    tf.keras.layers.Dense(20, kernel_regularizer=tf.keras.regularizers.l1_l2(0,0.001),activation="softmax")

])

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(0.01),
            metrics=["accuracy"])
model.fit(X_train, y_train, epochs=5, batch_size=64,validation_split=0.2,
         callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=2,verbose=0, mode='auto'))  


# In[125]:


tf.random.set_seed(123)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
    MaxTokens, 50, input_length = X_train.shape[1] ),
    tf.keras.layers.SpatialDropout1D(0.2),
    tf.keras.layers.LSTM(50, dropout=0.2,recurrent_dropout = 0.2),
    tf.keras.layers.Dense(20, kernel_regularizer=tf.keras.regularizers.l1_l2(0.001,0.001),activation="softmax")

])

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(0.01),
            metrics=["accuracy"])
model.fit(X_train, y_train, epochs=5, batch_size=64,validation_split=0.2,
         callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=2,verbose=0, mode='auto'))  


# In[130]:


tf.random.set_seed(123)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
    MaxTokens, 50, input_length = X_train.shape[1] ),
    tf.keras.layers.SpatialDropout1D(0.2),
    tf.keras.layers.LSTM(50, dropout=0.2,recurrent_dropout = 0.2),
    tf.keras.layers.Dense(20, kernel_regularizer=tf.keras.regularizers.l1_l2(0.01,0.1),activation="softmax")

])

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(0.1),
            metrics=["accuracy"])
model.fit(X_train, y_train, epochs=5, batch_size=64,validation_split=0.2,
         callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=2,verbose=0, mode='auto'))  


# In[139]:


tf.random.set_seed(123)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
    MaxTokens, 50, input_length = X_train.shape[1] ),
    tf.keras.layers.LSTM(50, dropout=0.3,recurrent_dropout = 0.3),
    tf.keras.layers.Dense(20, kernel_regularizer=tf.keras.regularizers.l1_l2(0.01,0.01),activation="softmax")

])

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(0.05),
            metrics=["accuracy"])
model.fit(X_train, y_train, epochs=5, batch_size=64,validation_split=0.2,
         callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=2,verbose=0, mode='auto')) 


# In[243]:


tf.random.set_seed(123)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
    MaxTokens, 20, input_length = X_train.shape[1] ),
    tf.keras.layers.SpatialDropout1D(0.25),
    tf.keras.layers.LSTM(20, dropout=0.2,recurrent_dropout = 0.2,return_sequences=True),
    tf.keras.layers.LSTM(20),
    tf.keras.layers.Dense(20, kernel_regularizer=tf.keras.regularizers.l1_l2(0,0),activation="softmax")

])

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(0.01),
            metrics=["accuracy"])
model.fit(X_train, y_train, epochs=5, batch_size=64,validation_split=0.2,
         callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=2,verbose=0, mode='auto')) 


# In[315]:


tf.random.set_seed(123)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
    MaxTokens, 100, input_length = X_train.shape[1] ),
    tf.keras.layers.SpatialDropout1D(0.2),
    tf.keras.layers.LSTM(100, dropout=0.2,recurrent_dropout = 0.2),
    tf.keras.layers.Dense(20, kernel_regularizer=tf.keras.regularizers.l1_l2(0,0),activation="softmax")

])

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(0.005),
            metrics=["accuracy"])
model.fit(X_train, y_train, epochs=5, batch_size=64,validation_split=0.2,
         callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=1,verbose=0, mode='auto'))


# In[ ]:


accr = model.evaluate(X_test,y_test) 


# # Best GRU Model

# In[356]:



tf.random.set_seed(123)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
    MaxTokens, 100, mask_zero=True, input_length=142 ),
    tf.keras.layers.SpatialDropout1D(0.25),
    tf.keras.layers.GRU(100),
    tf.keras.layers.Dense(20, kernel_regularizer=tf.keras.regularizers.l1_l2(0,0),activation="softmax")

])

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(0.001),
            metrics=["accuracy"])
model.summary()
model.fit(X_train, y_train, epochs=5, batch_size=64,validation_split=0.2,
         callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=1,verbose=0, mode='auto')) 


# In[357]:


accr = model.evaluate(X_test,y_test) 






