
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sys
import pickle


# In[2]:


# loading test data
test = pd.read_table(sys.argv[1],encoding='utf8')


# In[3]:


# data cleaning
word_test = test['id,text'].copy()
for i in range(len(word_test)):
    index = word_test[i].find(',')
    tmp = word_test[i][(index+1):]
    word_test[i] = tmp


# In[4]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[5]:


# loading tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


# In[6]:


word_len = 100
sequences = tokenizer.texts_to_sequences(word_test)
data_test = pad_sequences(sequences, maxlen=word_len)


# In[7]:


# loading model
from keras.models import load_model
model_glove = load_model("model.h5")


# In[8]:


# predict ans
test_label = model_glove.predict_classes(data_test)
test_label = test_label.flatten()


# In[9]:


# ans output
with open(sys.argv[2],'w',encoding='utf8') as f:
    f.write("id,label\n")
    for i in range(test_label.shape[0]):
        f.write("%s,%s\n"%(i,test_label[i]))

