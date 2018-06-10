
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sys

# In[2]:


test = pd.read_csv(sys.argv[1])


# In[3]:


import keras.backend as K


# In[4]:


def rmse(y_true, y_pred):
    y_pred = K.clip(y_pred, 1., 5.)
    return K.sqrt(K.mean(K.square((y_true - y_pred))))


# In[7]:


from keras.models import load_model
model = load_model('model.h5', custom_objects={'rmse': rmse})
prediction = model.predict([test.UserID.values,test.MovieID.values])

with open(sys.argv[2],'w') as f:
    f.write('TestDataID,Rating\n')
    for i in range(len(test)):
        f.write("%s,%s\n"%(i+1,float(prediction[i])))

