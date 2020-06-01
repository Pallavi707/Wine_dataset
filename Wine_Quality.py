#!/usr/bin/env python
# coding: utf-8

# # Red Wine Quality Test using classification through KNN - model

# In[33]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[34]:


#importing dataset of redwine qualities
X = pd.read_csv('red_wine.csv')

#Showing first five rows of the dataset
X.head()


# In[35]:


#Dataset has 1599 rows and 12 columns
X.shape


# In[36]:


X.columns.values


# In[37]:


#There are 12 columns out of which 11 are independent variables and one is dependent variable that is the quality variables.


# In[38]:


X.info()


# In[39]:


#Data has only float and int values and there are no missing values


# In[40]:


X.describe()
# Used to find out mean , meadian,std etc for each column as shown


# In[41]:


## 1)Mean values is greater than the median for each row and the difference is not much high except for the "total sulfur dioxide " column
#where there is a differnce in mean and median of about 8.467792

## 2)Notably large differences between 75% and max values

## 3)Above two observations indicate presence of outliers in the dataset


# In[42]:


#Understanding Target Variable


# In[43]:


X.quality.unique()


# In[44]:


# Target variable is discrete and categorical in nature 
# where the category value lies in the range of 1 to 10 where 1 means poor and 10 means best
# However in dataset the value of quality ranges from min value of 3 to max value of 8,(1,2,9,10 are not present)


# In[45]:


X.quality.value_counts()


# In[46]:


## It gives us the vote_counts for each quality values present in the dataset
## where a quality of 5 and 6 has maximum observations for quality
## ans 3,8 has minimum observations for quality
## That is mostly the observation performed is related to quality 5 and 6 wines and least number of observations are
##performed for quality values of 3 and 8

## It only refers to the no of observations available for each quality value


# In[47]:


df.isnull()                                                                                                                                                                        


# # Plotting graphs to check outliers

# In[48]:


# Using describe method and null method we already checked that there are no missing value


# In[49]:


# Plot to show presence of a missing value or not


# In[50]:


sns.heatmap(X.isnull(),cbar = False, yticklabels= False, cmap = 'viridis')


# In[51]:


#The graph also indicates that no missing values are present because if there was 
# then there must be a different color bar for that column name in which missing value is present


# In[52]:


## Checking for outliers


# In[53]:


col = X.columns.values
number_of_columns = 12
number_of_rows = len(col) - 1/ number_of_columns
plt.figure(figsize=(number_of_columns,5*number_of_rows))
for i in range(0,len(col)):
    plt.subplot(number_of_rows + 1,number_of_columns, i+1)
    sns.set_style('whitegrid')
    sns.boxplot(df[col[i]],color = 'olive',orient = 'v')
    plt.tight_layout()


# In[54]:


## The above plot shows each column has outlier values and also looking at the graph we can find out the value of the outliers


# In[55]:


## We will perform scaling to solve outliers problem


# # Performing knn algorithm without scaling for different values of k

# In[57]:



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, X.quality, test_size=0.3) # 70% training and 30% test



# In[64]:


from sklearn.neighbors import KNeighborsClassifier


#Create KNN Classifier for k = 5
knn = KNeighborsClassifier(n_neighbors=5)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[65]:


#Create KNN Classifier for k = 3
knn = KNeighborsClassifier(n_neighbors=3)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[63]:


#Create KNN Classifier for k = 1
knn = KNeighborsClassifier(n_neighbors=1)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[66]:


#Create KNN Classifier for k = 9
knn = KNeighborsClassifier(n_neighbors=9)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[67]:


#Create KNN Classifier for k = 2
knn = KNeighborsClassifier(n_neighbors=2)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[68]:


## From above data we can see that accuracy remains same for k=5,3 and it is different for k = 1,2
## Also as we move from 3 to 9 accuracy rate was almost similar nearby to 0.5
## But it mas max for k = 1 


# # Performing scaling and then performing same knn algorithm

# In[71]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled)

X_test_scaled = scaler.fit_transform(X_test)
X_test = pd.DataFrame(X_test_scaled)


# In[72]:


X_train.iloc[:5]


# In[ ]:


## After performing scaling we will again calculate the accuracy for the given dataset


# In[73]:


from sklearn.neighbors import KNeighborsClassifier


#Create KNN Classifier for k = 5
knn = KNeighborsClassifier(n_neighbors=5)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[74]:



#Create KNN Classifier for k = 1
knn = KNeighborsClassifier(n_neighbors=5)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[75]:


#Create KNN Classifier for k = 3
knn = KNeighborsClassifier(n_neighbors=5)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[76]:


## Here we can see once we performed scaling the accuracy for every value of k increased and becomes equal


# In[77]:


## Now we will vary the training dataset and performed the same function


# In[78]:


## Here I divided the training dataset and test dataset into equal ratios of 5o%
X_train, X_test, y_train, y_test = train_test_split(X, X.quality, test_size=0.5)

knn = KNeighborsClassifier(n_neighbors=5)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[79]:


knn = KNeighborsClassifier(n_neighbors=1)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[80]:


knn = KNeighborsClassifier(n_neighbors=9)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[81]:


## When we changed the ratios of training dataset and test dataset then here in this case we can see 
## that there is not much differences in the accuracy value however the accuracy value is decreased by around 0.01 for all values of k
## So accuracy was more when training set was 70% of the total dataset and is less when it is 50%


# In[82]:


## Plotting for different k neighbours


# In[86]:



X_train, X_test, y_train, y_test = train_test_split(X, X.quality, random_state = 0) # 70% training and 30% test
knnreg = KNeighborsClassifier(n_neighbors = 5).fit(X_train, y_train)

print(knnreg.predict(X_test))
print('R-squared test score: {: .3f}'
      .format(knnreg.score(X_test, y_test)))


# In[ ]:




