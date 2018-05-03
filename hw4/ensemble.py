
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.regularizers import L1L2
from keras import optimizers


# In[2]:


train_data = pd.read_csv('train.csv')


# In[3]:


Y = train_data['income'].map({" <=50K":0," >50K":1}).values
train_data.drop('income',axis=1,inplace=True)

train_data.age = train_data.age.astype(float)
train_data.fnlwgt = train_data.fnlwgt.astype(float)
train_data.education_num = train_data.education_num.astype(float)
train_data.hours_per_week = train_data.hours_per_week.astype(float)

train_data = pd.get_dummies(train_data, columns=[
    "workclass", "education", "marital_status", "occupation", "relationship",
    "race", "sex", "native_country",
])


# In[4]:


drop_out_list = ["native_country_ ?", "native_country_ Cambodia", "native_country_ Canada",
                 "native_country_ China", "native_country_ Columbia", "native_country_ Cuba", "native_country_ Dominican-Republic",
                 "native_country_ Ecuador", "native_country_ El-Salvador", "native_country_ England", "native_country_ France",
                 "native_country_ Germany", "native_country_ Greece", "native_country_ Guatemala", "native_country_ Haiti",
                 "native_country_ Holand-Netherlands","native_country_ Honduras","native_country_ Hong","native_country_ Hungary",
                 "native_country_ India","native_country_ Iran","native_country_ Ireland","native_country_ Italy","native_country_ Jamaica",
                 "native_country_ Japan","native_country_ Laos","native_country_ Mexico","native_country_ Nicaragua",
                 "native_country_ Outlying-US(Guam-USVI-etc)","native_country_ Peru","native_country_ Philippines","native_country_ Poland",
                 "native_country_ Portugal","native_country_ Puerto-Rico","native_country_ Scotland","native_country_ South",
                 "native_country_ Taiwan","native_country_ Thailand","native_country_ Trinadad&Tobago","native_country_ United-States",
                 "native_country_ Vietnam","native_country_ Yugoslavia",
                 
                 "relationship_ Husband","relationship_ Other-relative", "relationship_ Wife",
                 "relationship_ Not-in-family", "relationship_ Own-child", "relationship_ Unmarried",
                 
                 "education_ 10th","education_ 11th", "education_ 12th", "education_ 1st-4th", "education_ 5th-6th", 
                 "education_ 7th-8th", "education_ 9th", "education_ Assoc-acdm", "education_ Assoc-voc", 
                 "education_ Bachelors", "education_ Doctorate", "education_ HS-grad", "education_ Masters", 
                 "education_ Preschool", "education_ Prof-school", "education_ Some-college",
                 
                 "race_ Amer-Indian-Eskimo", "race_ Asian-Pac-Islander", "race_ Other",
                "race_ White", "race_ Black"
                ]


# In[5]:


df = pd.DataFrame(train_data)
df.drop(drop_out_list, axis=1, inplace=True,)


# In[6]:


scale = []
for col in df.columns:
    s = [col]
    maximun = df[col].max()
    minimum = df[col].min()
    if maximun != 1:
        m = df[col].mean()
        v = df[col].std()
        s.append(m)
        s.append(v)
        df[col] = (df[col]-m)/v
        scale.append(s)


# In[7]:


x_train = df.values
y_train = Y


# In[8]:


model_1 = Sequential()
model_1.add(Dense(len(x_train[0]), activation='relu', input_dim=len(x_train[0])))
model_1.add(Dense(1, activation='sigmoid'))
model_1.compile(optimizer=optimizers.Adagrad(lr=0.0123),
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])
model_1.summary()


# In[9]:


history1 = model_1.fit(x_train,y_train,epochs=50,batch_size=128,validation_split=0.1)


# In[10]:


model_2 = Sequential()
model_2.add(Dense(len(x_train[0]), activation='relu', input_dim=len(x_train[0])))
model_2.add(Dense(16))
model_2.add(Dense(1, activation='sigmoid'))
model_2.compile(optimizer=optimizers.Adagrad(lr=0.0123),
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])
model_2.summary()


# In[11]:


history2 = model_2.fit(x_train,y_train,epochs=50,batch_size=128,validation_split=0.1) 


# In[12]:


model_3 = Sequential()
model_3.add(Dense(len(x_train[0]), activation='relu', input_dim=len(x_train[0])))
model_3.add(BatchNormalization())
model_3.add(Dense(1, activation='sigmoid'))
model_3.compile(optimizer=optimizers.Adagrad(lr=0.01),
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])
model_3.summary()


# In[13]:


history3 = model_3.fit(x_train,y_train,epochs=60,batch_size=256,validation_split=0.1) 


# In[14]:


import matplotlib.pyplot as plt 
history3.history
plt.plot(history3.history['binary_accuracy'])
plt.plot(history3.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('model_3.png')
plt.show()


# In[15]:


from keras.models import save_model
save_model(model_1,'model_1.h5')
save_model(model_2,'model_2.h5')
save_model(model_3,'model_3.h5')


# In[16]:


test_data = pd.read_csv('test.csv')
test_data.age = test_data.age.astype(float)
test_data.fnlwgt = test_data.fnlwgt.astype(float)
test_data.education_num = test_data.education_num.astype(float)
test_data.hours_per_week = test_data.hours_per_week.astype(float)
test_data = pd.get_dummies(test_data, columns=[
    "workclass", "marital_status", "occupation", "sex"
])


# In[17]:


# Get missing columns in the training test
missing_cols = set( train_data.columns ) - set( test_data.columns )
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    test_data[c] = 0

test_df = pd.DataFrame(test_data)
test_df.drop(drop_out_list, axis=1, inplace=True,)
# Ensure the order of column in the test set is in the same order than in train set
test_df = test_df[df.columns]


# In[19]:


for s in scale:
    col = s[0]
    test_df[col] = (test_df[col]-s[1])/s[2]
    #test_df[col] = (test_df[col]-s[2])/(s[1]-s[2])
input_x = test_df.values
predict_1 = model_1.predict(input_x)
predict_2 = model_2.predict(input_x)
predict_3 = model_3.predict(input_x)


# In[22]:


result = []
for i in range(len(predict_1)):
    id = str(i+1)
    average = (predict_1[i]+predict_2[i]+predict_3[i])/3.0
    if average > 0.5:
        result.append([id, 1])
    else:
        result.append([id, 0])

export_df = pd.DataFrame(result,columns=['id','label'])
export_df.to_csv('ans.csv',sep=',',encoding='utf-8',index=False)

