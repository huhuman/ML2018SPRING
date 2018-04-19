
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import keras
from keras.layers import MaxPooling2D, Conv2D, Conv1D
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, advanced_activations
from keras.layers.merge import concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import sys

# In[10]:


model = load_model('hsu_model_v3.h5')


# In[4]:


#load training data
test_data = pd.read_csv(sys.argv[1])
test_data.feature = [test_data.feature.values[i].split(' ') for i in range(len(test_data))]
data_mean, data_std = np.load('norm.npy')


# In[6]:


x_test = []
for ele in test_data.feature.values:
    x_test.append(np.reshape(ele,(48,48,1)))
x_test = np.array(x_test).astype('float32')
x_test_norm = (x_test-data_mean)/data_std


# In[11]:


export_r = test_data.copy()
export_r.loc[:,'feature'] = model.predict_classes(x_test_norm)
export_r.columns = ['id','label']
export_r.to_csv(sys.argv[2],sep=',',encoding='utf-8',index=False)

