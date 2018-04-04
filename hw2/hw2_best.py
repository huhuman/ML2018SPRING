
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import L1L2
from keras import optimizers
import sys

# In[2]:


train_data = pd.read_csv(sys.argv[1])


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


# In[16]:


#wanted_list = [(col,abs(np.corrcoef(train_data[col].values,Y)[0,1])) for col in train_data.columns]
#wanted_list = np.flip(np.sort(np.array(wanted_list,dtype=[('colname','U100'),('corr',float)]),order='corr'),0)


# In[42]:


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


#wanted = [wanted_list[i][0] for i in range(30)]
#drop_out_list = set(train_data) - set(wanted)
#drop_out_list  = set(drop_out_list) - set(wanted)

df = pd.DataFrame(train_data)
df.drop(drop_out_list, axis=1, inplace=True,)


# In[43]:


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
        #df[col] = (df[col]-minimum)/maximun-minimum
        #s.append(maximun)
        #s.append(minimum)
        scale.append(s)


# In[44]:


"""
Prepare you data, such as:
"""
#all_index = random.sample(range(len(DataSet_Y)),len(DataSet_Y))
#train_index = all_index[:int(len(DataSet_Y)*0.9)]
#test_index = all_index[int(len(DataSet_Y)*0.9):]
#x_train = np.array([DataSet_X[i] for i in train_index],dtype=float)  # should be a numpy array
#y_train = np.array([DataSet_Y[i] for i in train_index],dtype=float)  # should be a numpy array
#x_val = np.array([DataSet_X[i] for i in test_index],dtype=float)    # should be a numpy array
#y_val = np.array([DataSet_Y[i] for i in test_index],dtype=float)    # should be a numpy array

x_train = df.values
y_train = Y

"""
Set up the logistic regression model
"""
model = Sequential()
#model.add(Dense(len(x_train[0]), kernel_initializer='normal', activation='sigmoid', input_dim=len(x_train[0])))
#model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
model.add(Dense(len(x_train[0]), activation='relu', input_dim=len(x_train[0])))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.Adagrad(lr=0.0123),
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])


# In[45]:


model.fit(x_train,y_train,epochs=50,batch_size=128,validation_split=0.1)


# In[46]:


#input data
test_data = pd.read_csv(sys.argv[2])
test_data.age = test_data.age.astype(float)
test_data.fnlwgt = test_data.fnlwgt.astype(float)
test_data.education_num = test_data.education_num.astype(float)
test_data.hours_per_week = test_data.hours_per_week.astype(float)
test_data = pd.get_dummies(test_data, columns=[
    "workclass", "marital_status", "occupation", "sex"
])

# Get missing columns in the training test
missing_cols = set( train_data.columns ) - set( test_data.columns )
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    test_data[c] = 0

test_df = pd.DataFrame(test_data)
test_df.drop(drop_out_list, axis=1, inplace=True,)
# Ensure the order of column in the test set is in the same order than in train set
test_df = test_df[df.columns]


# In[47]:


for s in scale:
    col = s[0]
    test_df[col] = (test_df[col]-s[1])/s[2]
    #test_df[col] = (test_df[col]-s[2])/(s[1]-s[2])
input_x = test_df.values
predict = model.predict(input_x)


# In[48]:


result = []
for i in range(len(predict)):
    id = str(i+1)
    if predict[i] > 0.5:
        result.append([id, 1])
    else:
        result.append([id, 0])

export_df = pd.DataFrame(result,columns=['id','label'])
export_df.to_csv(sys.argv[6],sep=',',encoding='utf-8',index=False)
