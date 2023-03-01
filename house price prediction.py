#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


customers = pd.read_csv('USA_Housing.csv')
customers.head()


# In[3]:


customers.describe()


# In[4]:


customers.info()


# In[5]:


sns.pairplot(customers)


# In[6]:


scaler = StandardScaler()

X=customers.drop(['Price','Address'],axis=1)
y=customers['Price']

cols = X.columns

X = scaler.fit_transform(X)


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[8]:


lr = LinearRegression()
lr.fit(X_train,y_train)

pred = lr.predict(X_test)

r2_score(y_test,pred)


# In[9]:


sns.scatterplot(x=y_test, y=pred)


# In[10]:


sns.histplot((y_test-pred),bins=50,kde=True)


# In[11]:


cdf=pd.DataFrame(lr.coef_, cols, ['coefficients']).sort_values('coefficients',ascending=False)
cdf


# In[ ]:




