#!/usr/bin/env python
# coding: utf-8

# In[171]:


import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')


# In[172]:


df= pd.read_csv("heart1.csv")


# In[173]:


df.head()


# In[174]:


df.tail()


# In[175]:


df.shape


# In[176]:


df.columns


# In[177]:


df.info()


# In[178]:


f,ax=plt.subplots(figsize=(10,10))
sns.heatmap(df.corr(),annot=True,linewidth=0.5,fmt=".1f",ax=ax)
plt.show()


# In[179]:


df.isnull().sum()


# In[180]:


df.describe()


# In[181]:


df['target'].value_counts()


# In[182]:


X = df.drop(columns='target',axis=1)
Y = df['target']


# In[183]:


print(X)


# In[184]:


print(Y)


# In[185]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)


# In[186]:


print(X.shape,X_train.shape,X_test.shape)


# In[187]:


model=LogisticRegression()


# In[188]:


model.fit(X_train,Y_train)


# In[189]:


model.predict(X_test)


# In[190]:


X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)


# In[191]:


print('Accuracy on Training data : ',training_data_accuracy)


# In[192]:


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)


# In[193]:


print('Accuracy on Test data : ',test_data_accuracy)


# In[194]:


y_prob=model.predict_proba(X_test)[:,1]
fpr,ftr,thresholds=roc_curve(Y_test,y_prob)
fig,ax=plt.subplots()
ax.plot(fpr,ftr)
ax.plot([0,1],[0,1],transform=ax.transAxes,ls="--",c=".3")
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.title('ROC curve ')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity) ')
plt.grid(True)


# In[195]:


input_data = (71,0,2,160,302,0,0,162,0,0.4,1,2,3)


input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)


# In[196]:


if (prediction[0]==0):
  print('The Person does not have a Herat Disease')
else:
  print('The Person has Herat Disease')


# In[197]:


input_data1 = (65,0,4,150,225,0,2,114,0,1,2,3,7)


input_data1_as_numpy_array = np.asarray(input_data1)

input_data1_reshaped = input_data1_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data1_reshaped)
print(prediction)
if (prediction[0]==0):
  print('The Person does not have a Herat Disease')
else:
  print('The Person has Herat Disease')


# In[198]:


model1=RandomForestClassifier(max_depth=5)
model1.fit(X_train,Y_train)


# In[199]:


model1.predict(X_test)


# In[200]:


X_train_prediction = model1.predict(X_train)
training1_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print('Accuracy on Training data : ',training1_data_accuracy)


# In[201]:


X_test_prediction = model1.predict(X_test)
test1_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print('Accuracy on Test data : ',test1_data_accuracy)


# In[202]:


y_prob=model1.predict_proba(X_test)[:,1]
fpr,ftr,thresholds=roc_curve(Y_test,y_prob)
fig,ax=plt.subplots()
ax.plot(fpr,ftr)
ax.plot([0,1],[0,1],transform=ax.transAxes,ls="--",c=".3")
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.title('ROC curve ')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity) ')
plt.grid(True)


# In[203]:


model2=SVC()
model2.fit(X_train,Y_train)


# In[204]:


model2.predict(X_test)


# In[205]:


X_train_prediction = model2.predict(X_train)
training2_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print('Accuracy on Training data : ',training2_data_accuracy)


# In[206]:


X_test_prediction = model2.predict(X_test)
test2_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print('Accuracy on Test data : ',test2_data_accuracy)


# In[207]:


model3=GradientBoostingClassifier()
model3.fit(X_train,Y_train)


# In[208]:


model3.predict(X_test)


# In[209]:


X_train_prediction = model3.predict(X_train)
training3_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print('Accuracy on Training data : ',training3_data_accuracy)


# In[210]:


X_test_prediction = model3.predict(X_test)
test3_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print('Accuracy on Test data : ',test3_data_accuracy)


# In[211]:


y_prob=model3.predict_proba(X_test)[:,1]
fpr,ftr,thresholds=roc_curve(Y_test,y_prob)
fig,ax=plt.subplots()
ax.plot(fpr,ftr)
ax.plot([0,1],[0,1],transform=ax.transAxes,ls="--",c=".3")
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
# plt.rcParams['font-size'] = 12
plt.title('ROC curve ')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity) ')
plt.grid(True)


# In[212]:


model4=GaussianNB()
model4.fit(X_train,Y_train)


# In[213]:


model4.predict(X_test)


# In[214]:


X_train_prediction = model4.predict(X_train)
training4_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print('Accuracy on Training data : ',training4_data_accuracy)


# In[215]:


X_test_prediction = model4.predict(X_test)
test4_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print('Accuracy on Test data : ',test4_data_accuracy)


# In[216]:


model5=KNeighborsClassifier()
model5.fit(X_train,Y_train)


# In[217]:


model5.predict(X_test)


# In[218]:


X_train_prediction = model5.predict(X_train)
training5_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print('Accuracy on Training data : ',training5_data_accuracy)


# In[219]:


X_test_prediction = model5.predict(X_test)
test5_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print('Accuracy on Test data : ',test5_data_accuracy)


# In[220]:


model6=DecisionTreeClassifier()
model6.fit(X_train,Y_train)


# In[221]:


model6.predict(X_test)


# In[222]:


X_train_prediction = model6.predict(X_train)
training6_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print('Accuracy on Training data : ',training6_data_accuracy)


# In[223]:


X_test_prediction = model6.predict(X_test)
test6_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print('Accuracy on Test data : ',test6_data_accuracy)


# In[224]:


kf=KFold(n_splits=14)
kf


# In[225]:


for train_index,test_index in kf.split(['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']):
    print(train_index,test_index)


# In[226]:


def get_score(model,X_train,X_test,Y_train,Y_test):
    model.fit(X_train,Y_train)
    return model.score(X_test,Y_test)


# In[227]:


get_score(LogisticRegression(),X_train,X_test,Y_train,Y_test)


# In[228]:


get_score(RandomForestClassifier(max_depth=5),X_train,X_test,Y_train,Y_test)


# In[229]:


get_score(SVC(),X_train,X_test,Y_train,Y_test)


# In[230]:


get_score(GradientBoostingClassifier(),X_train,X_test,Y_train,Y_test)


# In[231]:


get_score(GaussianNB(),X_train,X_test,Y_train,Y_test)


# In[232]:


get_score(KNeighborsClassifier(),X_train,X_test,Y_train,Y_test)


# In[233]:


get_score(DecisionTreeClassifier(),X_train,X_test,Y_train,Y_test)


# In[234]:


model_ev=pd.DataFrame({'Model':['Logistic Regression','Random Forest','Supprt Vector Machine','Stochastic Gradient Boosting','Naive Bayes','K Nearest Neighbors','Decision Tree'],'Accuracy':[test_data_accuracy*100,test1_data_accuracy*100,test2_data_accuracy*100,test3_data_accuracy*100,test4_data_accuracy*100,test5_data_accuracy*100,test6_data_accuracy*100]})
model_ev


# In[235]:


colors=['red','orange','yellow','green','blue','violet','purple']
plt.figure(figsize=(20,8))
plt.title("Barchart Accuracy of Different ML Models")
plt.xlabel("Algorithms")
plt.ylabel("%Accuracy")
plt.bar(model_ev['Model'],model_ev['Accuracy'],color=colors)
plt.show()


# In[236]:


import pickle
filename='model.pkl'
pickle.dump(model1,open(filename,'wb'))


# In[237]:


load_model=pickle.load(open(filename,'rb'))


# In[238]:


load_model.predict([[63,1,1,145,233,1,2,150,0,2.3,3,0,6]])


# In[239]:


load_model.predict([[67,1,4,160,286,0,2,108,1,1.5,2,3,3]])


# In[ ]:




