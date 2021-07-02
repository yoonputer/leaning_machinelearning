#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle


# In[2]:


pickle.load(open('./saves/favorite_save.pkl','rb'))


# In[3]:


favorite_load = pickle.load(open('./saves/favorite_save.pkl','rb'))
print(favorite_load)


# In[4]:


type(favorite_load)


# In[5]:


favorite_load['tiger']


# In[7]:


autompg_lr = pickle.load(open('./saves/autompg_lr.pkl','rb'))
print(autompg_lr)


# In[8]:


print(type(autompg_lr))


# In[9]:


autompg_lr.predict([[3504.0,8]])


# In[ ]:





# In[11]:

# input from outside
#-----------------------------------
a = 3504.0
b = 8

import numpy as np
pre = np.array([[a,b]])
print(autompg_lr.predict(pre))
#----------------↕ 의미가 같음-------- -----------
print(autompg_lr.predict([[3504.0,8]]))






# In[ ]:




