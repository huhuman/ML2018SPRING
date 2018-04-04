
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sys


# In[2]:


def sigmoid(z):
    return 1.0 / (1.0+np.exp(-1.0*z))

def Accuracy(x_data,y_data,w):
    count = 0
    for i in range(len(x_data)):
        f = sigmoid(np.dot(w.T,x_data[i]))
        if f > 0.5:
            if float(y_data[i])==1:
                count+=1
        else:
            if float(y_data[i])==0:
                count+=1
    return (1.0*count/len(y_data))

def Proba_Generative(x_data,y_data):
    N = len(y_data)
    n1 = np.count_nonzero(y_data)
    n2 = N-n1
    u1 = np.zeros(len(x_data[0]),dtype=float)
    u2 = np.zeros(len(x_data[0]),dtype=float)
    sigma1 = np.zeros((len(x_data[0]),len(x_data[0])),dtype=float)
    sigma2 = np.zeros((len(x_data[0]),len(x_data[0])),dtype=float)
    for i in range(len(x_data)):
        if y_data[i] == 0:
            u2 += x_data[i]
        elif y_data[i] == 1:
            u1 += x_data[i]
    u1/=n1
    u2/=n2
    for i in range(len(x_data)):
        if y_data[i] == 0:
            x = np.array(x_data[i],dtype=float)
            sigma2 += np.dot(np.transpose([x-u2]),[x-u2])
        elif y_data[i] == 1:
            x = np.array(x_data[i],dtype=float)
            sigma1 += np.dot(np.transpose([x-u1]),[x-u1])
    sigma = (sigma1+sigma2)/N
    inverse_sigma = np.linalg.inv(sigma)
    w = np.dot((u1-u2),inverse_sigma)
    b = np.log(n1/n2)+np.dot(np.dot(u2,inverse_sigma),u2.T)/2-np.dot(np.dot(u1,inverse_sigma),u1.T)/2
    return w,b

def test_output(x_data,w):
    result = []
    for i in range(len(x_data)):
        id = str(i+1)
        err = sigmoid(np.dot(w.T,x_data[i]))
        if err > 0.5:
            result.append([id, 1])
        else:
            result.append([id, 0])
    return result


# In[3]:


train_data = pd.read_csv(sys.argv[1])


# In[4]:


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


# In[5]:


drop_out_list = [#"capital_gain","capital_loss"
                 "occupation_ ?", "workclass_ ?",
                 
                 "native_country_ ?", "native_country_ Cambodia", "native_country_ Canada",
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
                 
                 "relationship_ Husband","relationship_ Not-in-family","relationship_ Other-relative",
                 "relationship_ Own-child","relationship_ Unmarried", "relationship_ Wife",
                 
                 "education_ 10th","education_ 11th", "education_ 12th", "education_ 1st-4th", "education_ 5th-6th", 
                 "education_ 7th-8th", "education_ 9th", "education_ Assoc-acdm", "education_ Assoc-voc", 
                 "education_ Bachelors", "education_ Doctorate", "education_ HS-grad", "education_ Masters", 
                 "education_ Preschool", "education_ Prof-school", "education_ Some-college",
                 
                 "race_ Amer-Indian-Eskimo", "race_ Asian-Pac-Islander", "race_ Black", "race_ Other", "race_ White"
                ]
df = pd.DataFrame(train_data)
df.drop(drop_out_list, axis=1, inplace=True,)


# In[6]:


scale = []
for col in df.columns:
    s = [col]
    maximun = max(df[col])
    minimum = min(df[col])
    if maximun != 1:
        m = np.mean(df[col])
        v = np.sqrt(np.var(df[col]))
        s.append(m)
        s.append(v)
        df[col] = (df[col]-m)/v
        #df[col] = (df[col]-minimum)/maximun-minimum
        #s.append(maximun)
        #s.append(minimum)
        scale.append(s)
DataSet_X = df.values
DataSet_Y = Y


# In[7]:


gener_w,gener_b = Proba_Generative(DataSet_X,DataSet_Y)


# In[11]:


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


# In[13]:


for s in scale:
    col = s[0]
    test_df[col] = (test_df[col]-s[1])/s[2]


# In[14]:


gener_input = []
for i in range(test_df.shape[0]):
    gener_input.append([1.0]+list(test_df.iloc[i,:]))
gener_r = test_output(gener_input,np.append(gener_b,gener_w))
export_df = pd.DataFrame(gener_r,columns=['id','label'])
export_df.to_csv(sys.argv[6],sep=',',encoding='utf-8',index=False)

