
# coding: utf-8

# In[2]:


import numpy as np
from skimage import io
import os, fnmatch
import sys

# In[6]:


img_path = sys.argv[1]
img_list = fnmatch.filter(os.listdir(img_path), '*.jpg')
All_imgs = []
for img in img_list:
    All_imgs.append(io.imread(img_path+img))
All_imgs = np.array(All_imgs)


# In[14]:


All_imgs = All_imgs.reshape(All_imgs.shape[0],-1)


# In[71]:


img_mean = np.mean(All_imgs,axis=0)
U, s, V = np.linalg.svd((All_imgs - img_mean).T, full_matrices=False)


# In[78]:


picked_img = img_path + '/' + sys.argv[2]


# In[82]:


four_eigenVectors = U.T[:4]
#reconstruction = []
img = picked_img
fig = plt.figure()
pic = io.imread('./Aberdeen/'+img)
pic = pic.flatten().astype('float32')
mu = np.mean(pic)
pic -= mu
recons = np.matmul(np.matmul(pic,four_eigenVectors.T),four_eigenVectors)+mu
recons -= np.min(recons)
recons /= np.max(recons)
recons = (recons*255).astype(np.uint8)
recons = recons.reshape(600,600,3)
#reconstruction.append(recons)
plt.title(img)
plt.savefig(img)

