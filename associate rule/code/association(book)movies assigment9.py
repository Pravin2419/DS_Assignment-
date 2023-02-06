#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[2]:


books=pd.read_csv("C:/Users/khushiagharkar/Downloads/book.csv")


# In[9]:


books


# In[10]:


books.info()


# # Apriori Algorithm 

# In[5]:


# at min_support=0.2 , support values will be more then 0.2
frequent_itemsets_2=apriori(books, min_support=0.2, use_colnames=True)
frequent_itemsets_2


# In[6]:


rules2 = association_rules(frequent_itemsets_2, metric="lift", min_threshold=0.7)
rules2


# In[11]:


rules2.sort_values('lift',ascending = False)[0:20]


# In[12]:


rules2[rules2.lift>1]


# In[13]:


df1=pd.DataFrame(data=frequent_itemsets_2)
df1
df1.duplicated()


# In[14]:


# visualization of obtained rule
rules2.plot(kind='bar',x='support',y='confidence',color='darkblue')
plt.title('Barplot')
plt.xlabel('support')
plt.ylabel('confidence')


# In[15]:


plt.scatter(rules2.support,rules2.confidence)
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# In[16]:


#at min_support=0.17 , support values will be greater then 0.17
frequent_itemsets_17=apriori(books, min_support=0.17, use_colnames=True)
frequent_itemsets_17


# In[17]:



rules17 = association_rules(frequent_itemsets_17, metric="lift", min_threshold=0.7)
rules17


# In[18]:


rules17.sort_values('lift',ascending = False)[0:20]


# In[19]:


rules17[rules17.lift>1]


# In[20]:


df17=pd.DataFrame(data=frequent_itemsets_17)
df17
df17.duplicated()


# In[22]:


rules17.plot(kind='bar',x='support',y='confidence',color='yellow')
plt.title('Barplot')
plt.xlabel('support')
plt.ylabel('confidence')


# In[23]:


plt.scatter(rules17.support,rules17.confidence)
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# In[24]:


#at min_support=0.15 ,support values will be greater then 0.17
frequent_itemsets_15=apriori(books, min_support=0.15, use_colnames=True)
frequent_itemsets_15


# In[25]:


rules15 = association_rules(frequent_itemsets_15, metric="lift", min_threshold=0.7)
rules15


# In[26]:


rules15.sort_values('lift',ascending = False)[0:20]


# In[27]:


df15=pd.DataFrame(data=frequent_itemsets_15)
df15
df15.duplicated()


# In[29]:


rules15.plot(kind='bar',x='support',y='confidence',color='purple')
plt.title('Barplot')
plt.xlabel('support')
plt.ylabel('confidence')
plt.figure(figsize=(20,10))


# In[30]:


plt.scatter(rules15.support,rules15.confidence)
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# In[31]:


x=[0.15,0.17,0.2]
y=[21,9,2]
plt.scatter(x,y)
plt.xlabel('Minimum Support')
plt.ylabel('Frequent Itemsets')
plt.title('Relation Between Min Support Value and Frequent Itemsets')


# In[32]:


x=[0.1,0.2,0.3]
y=[52,12,6]
plt.scatter(x,y)
plt.xlabel('Minimum Support')
plt.ylabel('Frequent Itemsets')
plt.title('Relation Between Min Support Value and Frequent Itemsets')


# In[ ]:




