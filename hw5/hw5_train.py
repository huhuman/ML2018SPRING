
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sys


# In[2]:


train = pd.read_table(sys.argv[1],header=None,sep="\+\+\+\$\+\+\+",encoding='utf8')
train_no_label = pd.read_table(sys.argv[2],header=None,sep="\+\+\+\$\+\+\+",encoding='utf8')


# In[3]:


word = train[1].copy()
word_nl = train_no_label[0].copy()


# In[4]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding


# In[5]:


#ref https://medium.com/@sabber/classifying-yelp-review-comments-using-cnn-lstm-and-pre-trained-glove-word-embeddings-part-3-53fcea9a17fa
word_len = 100

#vocabulary_size = len(word)
vocabulary_size = 40000

tokenizer = Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(word.values)
sequences = tokenizer.texts_to_sequences(word.values)
data = pad_sequences(sequences, maxlen=word_len)


# In[6]:


#ref https://medium.com/@sabber/classifying-yelp-review-comments-using-cnn-lstm-and-pre-trained-glove-word-embeddings-part-3-53fcea9a17fa
GLOVE_VECTOR_FILE = 'vectors_all_300.txt'
vector_dim = 300

embeddings_index = dict()
f = open(GLOVE_VECTOR_FILE,encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((vocabulary_size, vector_dim))
for word, index in tokenizer.word_index.items():
    if index > vocabulary_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector


# In[7]:


#ref https://medium.com/@sabber/classifying-yelp-review-comments-using-cnn-lstm-and-pre-trained-glove-word-embeddings-part-3-53fcea9a17fa
model_glove = Sequential()
model_glove.add(Embedding(len(embedding_matrix), vector_dim, input_length=word_len, weights=[embedding_matrix], trainable=False))
model_glove.add(LSTM(128,activation='tanh', dropout= 0.2, recurrent_dropout = 0.2))
model_glove.add(Dense(1, activation='sigmoid'))
model_glove.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
model_glove.summary()


# In[8]:


from keras import callbacks
import os
import datetime
curr = str(datetime.datetime.now())
curr = curr.split(' ')
save_path = 'model/%s_%s_model.h5'%("".join(curr[0].split('-')[1:]),"".join(curr[1].split(":")[0:2]))

earlystopping = callbacks.EarlyStopping(monitor='val_acc', patience = 4, verbose=1, mode='max')
checkpoint = callbacks.ModelCheckpoint(filepath = save_path,
                                       verbose = 1,
                                       save_best_only = True,
                                       save_weights_only = False,
                                       monitor = 'val_acc',
                                       mode='max')


# In[9]:


train_history = model_glove.fit(data, train[0].values,
                                validation_split=0.1,
                                epochs = 20,
                                batch_size=256, 
                                verbose=1,
                                callbacks=[checkpoint, earlystopping])


# In[ ]:


from keras.models import load_model
model_glove = load_model(save_path)


# In[ ]:


sequences = tokenizer.texts_to_sequences(word_nl)
data_nl = pad_sequences(sequences, maxlen=word_len)


# In[ ]:


thres = 0.8

nl_result = model_glove.predict(data_nl)
nl_result = nl_result.flatten()
nl_Positive = data_nl[nl_result > thres].copy()
nl_Negative = data_nl[nl_result < (1-thres)].copy()


# In[ ]:


semi_label = np.append(np.repeat(1,nl_Positive.shape[0]),np.repeat(0,nl_Negative.shape[0]))
semi_data = np.concatenate((nl_Positive,nl_Negative))


# In[ ]:


new_data = np.concatenate((data,semi_data))
new_label = np.append(train[0].values,semi_label)


# In[ ]:


#ref https://medium.com/@sabber/classifying-yelp-review-comments-using-cnn-lstm-and-pre-trained-glove-word-embeddings-part-3-53fcea9a17fa
model_glove = Sequential()
model_glove.add(Embedding(len(embedding_matrix), vector_dim, input_length=word_len, weights=[embedding_matrix], trainable=False))
model_glove.add(LSTM(128,activation='tanh', dropout= 0.2, recurrent_dropout = 0.2))
model_glove.add(Dense(1, activation='sigmoid'))
model_glove.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
model_glove.summary()


# In[ ]:


save_path = 'model/%s_%s_semimodel_%s.h5'%("".join(curr[0].split('-')[1:]),"".join(curr[1].split(":")[0:2]),thres)

earlystopping = callbacks.EarlyStopping(monitor='val_acc', patience = 2, verbose=1, mode='max')
checkpoint = callbacks.ModelCheckpoint(filepath = save_path,
                                       verbose = 1,
                                       save_best_only = True,
                                       save_weights_only = False, 
                                       monitor = 'val_acc',
                                       mode='max')


# In[ ]:


train_history = model_glove.fit(new_data, new_label,
                                validation_split=0.1, 
                                epochs = 15, 
                                batch_size=256,
                                verbose=1, 
                                callbacks=[checkpoint, earlystopping])

