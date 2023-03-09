#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install pandas')


# In[3]:


get_ipython().system('pip install numpy')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install sklearn')


# In[4]:


import numpy as np 
import pandas as pd 
import seaborn as sns 
import  matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression 

sns.set_style('darkgrid')


# In[6]:


## Dataset 
df = pd.read_csv('payment_fraud.csv')


# In[7]:


df.head()


# In[8]:


df.isnull().sum() ## checking the null valeus 


# In[9]:


paymthd = df.paymentMethod.value_counts()
plt.figure(figsize=(5, 5))
sns.barplot(paymthd.index, paymthd);
plt.ylabel('Count');


# In[10]:


df.label.value_counts() ## count the number of 0's and 1's


# In[11]:


## coverting paymentMethod column into label encoding
paymthd_label = {v:k for k, v in enumerate(df.paymentMethod.unique())}

df.paymentMethod = df.paymentMethod.map(paymthd_label)


# In[12]:


df.head()


# In[13]:


## corr(): it gives the correlation between the featuers
plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True);


# In[14]:


df.describe()


# In[15]:


## independent and dependent features
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


# In[16]:


## scaling 

sc = StandardScaler()
X = sc.fit_transform(X)


# In[17]:


## train test split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[18]:


print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)


# In[19]:


## logisticRegression Model
lg = LogisticRegression()

## training
lg.fit(X_train, y_train)


# In[20]:


## prediction 
pred = lg.predict(X_test)


# In[21]:


print("Accuracy")
print(accuracy_score(y_test, pred))
print()

print("Classification Report")
print(classification_report(y_test, pred))
print()

print("Confustion Metrics")
plt.figure(figsize=(10, 10));
sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='g');


# In[ ]:




