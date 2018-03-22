
# coding: utf-8

# In[5]:


import sys
import numpy as np
import pandas as pd
import scipy as sp


# In[2]:


#load csv file
def load_data(filename):
    return pd.read_csv(filename, encoding= 'big5')

def Get_Result_Scaling(w,input_data,scale):
    result = []
    for i in range(len(input_data)):
        f = np.array(input_data[i][1],dtype=float)
        for j in range(len(input_data[i][1])):
            if scale[j][0] != scale[j][1]:
                f[j] = (f[j]-scale[j][1])/(scale[j][0] - scale[j][1])
        result.append([input_data[i][0],np.dot(f,w)])
    return result


# In[6]:


test_data = load_data(sys.argv[1])
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


# In[7]:


filename = sys.argv[2]
w = np.load('model.npy')
scale = np.load('scale.npy')
result = Get_Result_Scaling(w,input_data,scale)
export_df = pd.DataFrame(result,columns=['id','value'])
export_df.to_csv(filename,sep=',',encoding='utf-8',index=False)

