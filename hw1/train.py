
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt


# In[2]:


#load csv file
def load_data(filename):
    return pd.read_csv(filename, encoding= 'big5')


# In[29]:


def Gradient_Descent(iteration, eta, x, y, lda = 0):
    Cost_Process = []
    X = np.array(x,dtype=float)
    Y = np.array(y,dtype=float)
    w = np.array([0]*X.shape[1],dtype=float).T
    s_grad = np.zeros(len(X[0]))
    for i in range(iteration):
        loss = np.dot(X,w) - Y
        gradient = np.dot(X.T,loss)/len(X)
        s_grad += gradient**2
        ada = np.sqrt(s_grad)
        #regularization
        w = w - eta*(gradient+lda*w)/ada
        if i%(100) == 0:    
            cost = np.sqrt(np.sum(loss**2)/len(X))
            Cost_Process.append(cost)
            if i%(iteration/10) == 0:
                print("Iteration %d | Cost: %f" % (i, cost))
    #Make_Figure_1D(Cost_Process,eta)
    return w


# In[8]:


def loss_function(w,X,Y):
    loss = np.dot(X,w) - Y
    return np.dot(X.T,loss)/len(X)


# In[5]:


def Get_Result(w,input_data):
    result = []
    for i in range(len(input_data)):
        result.append([input_data[i][0],np.dot(np.array(input_data[i][1],dtype=float),w)])
    return result

def Get_Result_Scaling(w,input_data,scale):
    result = []
    for i in range(len(input_data)):
        f = np.array(input_data[i][1],dtype=float)
        for j in range(len(input_data[i][1])):
            if scale[j][0] != scale[j][1]:
                f[j] = (f[j]-scale[j][1])/(scale[j][0] - scale[j][1])
        result.append([input_data[i][0],np.dot(f,w)])
    return result

def Make_Figure_1D(data,eta):
    plt.plot(data)
    T = "Learning rate " + str(eta)
    plt.title = T
    plt.ylabel = "Cost"
    plt.savefig("LR(" + str(eta) + ").png")
    plt.show()


# In[6]:


train_data = load_data("train.csv")


# In[9]:


#preprocess
p1 = train_data.iloc[:,[i for i in range(2,27)]]

each_Feature = lambda df, i, n: df.iloc[18*n:18*n+18].stack()[18*n+i]
create_df = lambda df: pd.DataFrame(df[1:25], columns=[df[0]])

final_data = pd.DataFrame(columns=p1.iloc[0:18,0])
for i in range(240):
    p2 = create_df(each_Feature(p1,0,i))
    for j in range(1,18):
        p2 = pd.concat([p2, create_df(each_Feature(p1,j,i))], axis=1)
    final_data = pd.concat([final_data, p2])
final_data = final_data.reset_index().reset_index().drop(columns = 'index')
final_data = final_data.rename(columns={'level_0':'time'})
final_data['RAINFALL'] = [0 if x=='NR' else x for x in final_data['RAINFALL']]


# In[10]:


def feature_scaling(data):
    features = [[ele[i] for ele in data] for i in range(len(data[0])) ]
    scaling_data = []
    scale = []
    for feature in features:
        f = np.array(feature,dtype=float)
        maximum = max(f)
        minimum = min(f)
        scale.append([maximum,minimum])
        if maximum != minimum:
            scaling_data.append([(ele-minimum)/(maximum-minimum) for ele in f])
        else:
            scaling_data.append(f)
    return [[scaling_data[i][j] for i in range(len(scaling_data))]for j in range(len(scaling_data[0])) ], scale


# In[9]:


#Create dataset of PM2.5
Seq = list(final_data['PM2.5'])
DataSet_X = []
DataSet_Y = []
for i in range(len(Seq)-9):
    x = [1.0] + Seq[i:i+9]
    y = Seq[i+9]
    DataSet_X.append(x)
    DataSet_Y.append(y)


# In[10]:


#Create dataset of all parameters in 9 hours
Seq2 = []
DataSet_X = []
DataSet_Y = []
for i in range(18):
    Seq2.append(list(final_data.iloc[:,i+1]))
for i in range(len(final_data)-9):
    temp = [1.0]
    for j in range(18):
        temp += Seq2[j][i:i+9]
    DataSet_X.append(temp)
    DataSet_Y.append(Seq2[9][i+9])


# In[11]:


#Create dataset of all parameters in an hour
Seq3 = []
DataSet_X = []
DataSet_Y = []
for i in range(18):
    Seq3.append(list(final_data.iloc[:,i+1]))
for i in range(len(Seq3[0])-1):
    if (i+1)%480 == 0:
        continue
    tmp = [1.0]
    for j in range(18):
        tmp += [Seq3[j][i]]
    if sum(ele!='0' for ele in tmp) != 1:
        DataSet_X.append(tmp)
        DataSet_Y.append(Seq3[9][i+1])


# In[52]:


'''
w = np.array([0]*10,dtype=float).T
X = np.array(DataSet_X,dtype=float)
Y = np.array(DataSet_Y,dtype=float)
loss = np.dot(X,w) - Y
gradient = np.dot(loss,X)/len(X)
w = w - 0.01*gradient
'''
#w2 = Adagrad_Gradient_Descent(100000, loss_function, DataSet_X, DataSet_Y, 1e-6, 0.0123)
f_DataSet_X, scale = feature_scaling(DataSet_X)
w = Gradient_Descent(50000, 4.123,f_DataSet_X,DataSet_Y)


# In[17]:


test_data = load_data("test.csv")


# In[51]:


#Get all PM2.5 related data
input_data = test_data.iloc[8:len(test_data):18].iloc[:,[0,2,3,4,5,6,7,8,9,10]]


# In[84]:


#Get all parameters in 9 hours
input_data = []
tmp = [1.0] + list(test_data.columns)[2:]
for i in range(17):
    tmp += [ele for ele in list(test_data.iloc[i][2:])]
tmp = [0 if x == 'NR' else x for x in tmp]
input_data.append(['id_0',tmp])
for j in range(1,260):
    tmp = [1.0]
    for i in range(18*j-1,18*j+17):
        tmp += [ele for ele in list(test_data.iloc[i][2:])]
    tmp = [0 if x == 'NR' else x for x in tmp]
    input_data.append([test_data.iloc[18*j-1][0], tmp])


# In[18]:


#in an hour
input_data = []
tmp = [1.0] + [list(test_data.columns)[-1]]
for i in range(17):
    tmp += [list(test_data.iloc[i])[-1]]
tmp = [0 if x == 'NR' else x for x in tmp]
input_data.append(['id_0',tmp])
for j in range(1,260):
    tmp = [1.0]
    for i in range(18*j-1,18*j+17):
        tmp += [list(test_data.iloc[i])[-1]]
    tmp = [0 if x == 'NR' else x for x in tmp]
    input_data.append([test_data.iloc[18*j-1][0], tmp])


# In[46]:


#result = Get_Result_Scaling(w,input_data,scale)
result = Get_Result(w,input_data)


# In[47]:


export_df = pd.DataFrame(result,columns=['id','value'])
export_df.to_csv("result_list.csv",sep=',',encoding='utf-8',index=False)


# In[53]:


np.save('model.npy',w)

