
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sys

# In[2]:


test = pd.read_csv(sys.argv[1])


# In[3]:


import keras.models as kmodels
import keras.layers as klayers
from keras.layers import Input, Embedding, Dense, Dropout, dot
from keras.layers import Reshape, Flatten, Lambda
from keras.callbacks import EarlyStopping,ModelCheckpoint
import keras.backend as K
import keras


# In[4]:


def rmse(y_true, y_pred):
    y_pred = K.clip(y_pred, 1., 5.)
    return K.sqrt(K.mean(K.square((y_true - y_pred))))


# In[7]:


from keras.models import load_model
model = load_model('model/0602_1612_model.h5', custom_objects={'rmse': rmse})
prediction = model.predict([test.UserID.values,test.MovieID.values])

with open(sys.argv[2],'w') as f:
    f.write('TestDataID,Rating\n')
    for i in range(len(test)):
        f.write("%s,%s\n"%(i+1,float(prediction[i])))

