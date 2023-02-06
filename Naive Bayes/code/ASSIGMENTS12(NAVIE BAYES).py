#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB


# In[2]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score


# In[3]:


test=pd.read_csv("C:/Users/khushiagharkar/Downloads/SalaryData_Test.csv")
train=pd.read_csv("C:/Users/khushiagharkar/Downloads\SalaryData_Train.csv")


# In[5]:


test.head()


# In[6]:


train.head()


# In[8]:


test.shape


# In[10]:


train.shape


# In[11]:


test.info()


# In[12]:


train.info()


# In[13]:



#Converting dtypes for train
train['workclass']=train['workclass'].astype('category')
train['education']=train['education'].astype('category')
train['maritalstatus']=train['maritalstatus'].astype('category')
train['occupation']=train['occupation'].astype('category')
train['relationship']=train['relationship'].astype('category')
train['race']=train['race'].astype('category')
train['native']=train['native'].astype('category')
train['sex']=train['sex'].astype('category')


# In[14]:


train.info()


# In[15]:


from sklearn import preprocessing                      
label_encoder = preprocessing.LabelEncoder()


# In[16]:


train['workclass'] = label_encoder.fit_transform(train['workclass'])
train['education'] = label_encoder.fit_transform(train['education'])
train['maritalstatus'] = label_encoder.fit_transform(train['maritalstatus'])
train['occupation'] = label_encoder.fit_transform(train['occupation'])
train['relationship'] = label_encoder.fit_transform(train['relationship'])
train['race'] = label_encoder.fit_transform(train['race'])
train['sex'] = label_encoder.fit_transform(train['sex'])
train['native'] = label_encoder.fit_transform(train['native'])


# In[17]:


train.head()


# In[18]:


test['workclass']=test['workclass'].astype('category')
test['education']=test['education'].astype('category')
test['maritalstatus']=test['maritalstatus'].astype('category')
test['occupation']=test['occupation'].astype('category')
test['relationship']=test['relationship'].astype('category')
test['race']=test['race'].astype('category')
test['native']=test['native'].astype('category')
test['sex']=test['sex'].astype('category')


# In[19]:


test.info()


# In[20]:


test['workclass'] = label_encoder.fit_transform(test['workclass'])
test['education'] = label_encoder.fit_transform(test['education'])
test['maritalstatus'] = label_encoder.fit_transform(test['maritalstatus'])
test['occupation'] = label_encoder.fit_transform(test['occupation'])
test['relationship'] = label_encoder.fit_transform(test['relationship'])
test['race'] = label_encoder.fit_transform(test['race'])
test['sex'] = label_encoder.fit_transform(test['sex'])
test['native'] = label_encoder.fit_transform(test['native'])


# In[21]:


test.head()


# In[24]:


train['Salary'] = label_encoder.fit_transform(train['Salary'])
test['Salary'] = label_encoder.fit_transform(test['Salary'])


# In[25]:


test.head()


# In[26]:


train.head()


# # Splitting the data into x train y train and x test y test 

# In[27]:


trainx=train.iloc[:,0:13]
trainy=train.iloc[:,13]
testx=test.iloc[:,0:13]
testy=test.iloc[:,13]


# In[28]:


trainx.shape ,trainy.shape, testx.shape, testy.shape


# In[29]:


# train a Gaussian Naive Bayes classifier on the training set
from sklearn.naive_bayes import GaussianNB


# In[30]:


gnb = GaussianNB()


# In[31]:


gnb.fit(trainx,trainy)


# In[32]:


y_pred = gnb.predict(testx)

y_pred


# In[33]:


#comparing train set and test set accuracing 
acc = accuracy_score(testy, y_pred) * 100
print("Accuracy =", acc)


# In[34]:


print('Training set score: {:.3f}'.format(gnb.score(trainx, trainy)))

print('Test set score: {:.3f}'.format(gnb.score(testx, testy)))


# # confusion matrix

# In[35]:


pd.crosstab(y_pred,testy)


# In[36]:


y_pred


# In[37]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(testy, y_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


# In[38]:


cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


# In[ ]:




