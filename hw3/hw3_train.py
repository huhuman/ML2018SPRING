
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import keras
from keras.layers import MaxPooling2D, Conv2D, Conv1D
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, advanced_activations
from keras.layers.merge import concatenate
from keras.preprocessing.image import ImageDataGenerator
import sys

# In[2]:


#training parameters
batch_size = 128
epochs = 200
class_num = 7


# In[4]:


#load training data
train_data = pd.read_csv(sys.argv[1])
train_data.feature = [train_data.feature.values[i].split(' ') for i in range(len(train_data))]


# In[5]:


x_train = []
for ele in train_data.feature.values:
    x_train.append(np.reshape(ele,(48,48,1)))
x_train = np.array(x_train).astype('float32')
#x_train = train_data.feature.values
y_train = np.reshape(train_data.label.values,(28709,1))
y_train = keras.utils.to_categorical(y_train,class_num)


# In[3]:


data_mean, data_std = np.load('norm.npy')


# In[7]:


x_train_norm = (x_train-data_mean)/data_std


# In[27]:


input1 = keras.layers.Input(shape=(48,48,1))
x1 = Conv2D(64,(3,3))(input1) 
x1 = advanced_activations.LeakyReLU(alpha=0.05)(x1)
x1 = Conv2D(64,(3,3))(x1) 
x1 = advanced_activations.LeakyReLU(alpha=0.05)(x1)
x1 = MaxPooling2D(pool_size=(2,2))(x1)
x1 = Dropout(0.1)(x1)

x1 = Conv2D(128,(3,3))(x1) 
x1 = advanced_activations.LeakyReLU(alpha=0.05)(x1) 
x1 = Conv2D(128,(3,3))(x1) 
x1 = advanced_activations.LeakyReLU(alpha=0.05)(x1)
x1 = MaxPooling2D(pool_size=(2,2))(x1)
x1 = Dropout(0.1)(x1)

x1 = Conv2D(256,(3,3))(x1) 
x1 = advanced_activations.LeakyReLU(alpha=0.05)(x1) 
x1 = Conv2D(256,(3,3))(x1) 
x1 = advanced_activations.LeakyReLU(alpha=0.05)(x1)
x1 = MaxPooling2D(pool_size=(2,2))(x1)
x1 = Dropout(0.4)(x1)

out = Flatten()(x1)
out = keras.layers.Dense(2048)(out)
out = Dropout(0.5)(out)
out = Dense(7,activation='softmax')(out)

model = keras.models.Model(inputs=[input1], outputs=out)


# In[28]:


datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=30,
    zoom_range = 0.25,
    horizontal_flip=True)


# In[29]:


model.compile(loss = 'categorical_crossentropy',
              optimizer = keras.optimizers.Adam(lr=0.001), 
              metrics = ['accuracy'])


# In[30]:


# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=round(len(x_train)/batch_size), epochs=epochs)

