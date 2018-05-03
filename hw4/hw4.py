
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sys

# In[2]:


X = np.load(sys.argv[1])
X = X.astype('float32')/255.
X = np.reshape(X,(len(X),-1))
x_train = X.copy()
x_train.shape


# In[4]:


from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='relu')(decoded)

encoder = Model(input=input_img, output= encoded)

adam = Adam(lr=5e-4)
autoencoder = Model(input=input_img, output=decoded)
autoencoder.compile(optimizer=adam,loss='mse')
autoencoder.summary()


# In[6]:


autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True)
#autoencoder.save('autoencoder_%02.0f%02.0f.h5'%(now.month,now.day))
#encoder.save('encoder_%02.0f%02.0f.h5'%(now.month,now.day))


# In[18]:


#from keras.models import load_model
#encoder = load_model('encoder_0503.h5')
from sklearn.cluster import KMeans
encoded_imgs = encoder.predict(X)
encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0], -1)
km = KMeans(n_clusters=2, random_state=0).fit(encoded_imgs)


# In[19]:


import csv
f = pd.read_csv(sys.argv[2])
o = open(sys.argv[3], 'w', newline="")
with o:
    writer = csv.writer(o)
    writer.writerow(['ID','Ans'])
    for i in range(len(f)):
        index1 = f.iloc[i,1]
        index2 = f.iloc[i,2]
        writer.writerow([i,int(km.labels_[index1]==km.labels_[index2])])

