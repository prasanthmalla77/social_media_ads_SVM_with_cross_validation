#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[5]:


dataset = pd.read_csv(r"C:\Users\91949\Downloads\Social_Network_Ads.csv")


# In[6]:


dataset.head()


# In[7]:


dataset.isna().sum()


# In[10]:


x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:,-1].values


# In[11]:


x[0]


# In[12]:


y


# In[13]:


#binary classification


# In[14]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[15]:


x_train.shape


# In[16]:


x_test.shape


# In[18]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train,y_train)


# In[19]:


y_pred = classifier.predict(x_test)


# In[20]:


y_pred


# In[22]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)


# In[23]:


accuracy_score(y_test,y_pred)


# In[27]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier , X= x_train, y = y_train, cv=10)


# In[28]:


accuracies


# In[30]:


accuracies.mean()


# In[31]:


accuracies.std()


# In[ ]:




