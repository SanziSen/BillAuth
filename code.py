#!/usr/bin/env python
# coding: utf-8

# In[23]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[24]:


dataset=pd.read_csv("bill_authentication.csv")


# In[25]:


dataset


# In[26]:


dataset.shape


# In[27]:


dataset.head()


# In[35]:


X=dataset.drop('Class',axis=1)
y=dataset['Class']


# In[36]:


X


# In[37]:


y


# In[38]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)


# In[ ]:





# In[39]:


X_train


# In[40]:


y_train


# In[41]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)


# In[42]:


y_pred = classifier.predict(X_test)


# In[43]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[50]:


from sklearn import metrics
print('Mean Absolute Error :',metrics.mean_absolute_error(y_test,y_pred) )
print('Mean Squared Error : ',metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error : ',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[44]:


from sklearn.linear_model import LogisticRegression


# In[45]:


model = LogisticRegression(solver='lbfgs', max_iter=2000, multi_class='auto')


# In[46]:


model.fit(X_train,y_train)


# In[47]:


model.score(X_test,y_test)


# In[ ]:




