#!/usr/bin/env python
# coding: utf-8

# ### Diabetes Prediction Model using Stacking Algorithm

# In[ ]:





# In[1]:


# importing the data.

import pandas as pd 
diabetes_data = pd.read_csv(r"C:\Users\DELL\Documents\MACHINE LEARNING\diabetes.csv")
print(diabetes_data)


# In[2]:


# to check whether data is balanced or not?

diabetes_data['Outcome'].value_counts()


# Hence, it's an imbalanced dataset, imbalanced towards class 0 i.e. No diabetes cases. So now we'll perform a sampling technique to balance the dataset.

# In[3]:


yes_diabetes=diabetes_data[diabetes_data.Outcome==1]
yes_diabetes.head()


# In[4]:


yes_diabetes.shape


# In[5]:


no_diabetes=diabetes_data[diabetes_data.Outcome==0]
no_diabetes.head()


# In[6]:


no_diabetes.shape


# In[7]:


# upsampling the minority class i.e. outcome=1 cases.

from sklearn.utils import resample 
yes_diabetes_upsampled = resample(yes_diabetes, replace=True, n_samples=470)
yes_diabetes_upsampled.shape


# In[33]:


# combining the upsampled data with no diabetes case from original data.

diabetes_data_new=pd.concat([yes_diabetes_upsampled,no_diabetes])
diabetes_data_new.shape


# In[37]:


# Shuffling the new combined data.

from sklearn.utils import shuffle
diabetes_data_new=shuffle(diabetes_data_new)
diabetes_data_new.head()


# In[38]:


# Selecting target variable

y = diabetes_data_new['Outcome']
y.head()


# In[39]:


# Selecting features.

x = diabetes_data_new.drop('Outcome', axis=1)
x.head()


# In[40]:


#splitting the data into training and testing.

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state=42)
x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[41]:


# creating stacking model.

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score


# Defining base estimators.
ada = AdaBoostClassifier(algorithm='SAMME',n_estimators=50, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Defining final estimator.
dt = DecisionTreeClassifier(random_state=42)

# Defining the stacking classifier model.
stacking_model = StackingClassifier(
    estimators=[('ada', ada), ('rf', rf)],
    final_estimator=dt,
    cv=5
)

# Training the stacking classifier model.
stacking_model.fit(x_train, y_train)

# Making predictions.
y_pred = stacking_model.predict(x_test)

# Generating classification report
print('STACKING MODEL - CLASSIFICATION REPORT\n')
report = classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes'])
print(report)
cm=confusion_matrix(y_test,y_pred)
print('STACKING MODEL - CONFUSION MATRIX\n')
print(cm)
print('')
score=roc_auc_score(y_test,y_pred)
print('STACKING MODEL - ROC AUC SCORE: ',score)


# In[44]:


# Visualization of confusion matrix.

import matplotlib.pyplot as plt
import seaborn as sns

cm=confusion_matrix(y_test,y_pred)

labels = ['No Diabetes', 'Diabetes']
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix for Stacking Model\n')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[ ]:





# In[76]:


# Plotting ROC Curve.

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

y_pred = stacking_model.predict(x_test)

fpr, tpr,_= roc_curve(y_test, y_pred)
plt.plot(fpr,tpr);
plt.plot([0,1],[0,1],linestyle='--');
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend(['ROC Curve'],loc='lower right')
plt.show()


# ## Analysis.

# ### Analysis of Classification report.
# 
# #### An accuracy of 0.81 means that 81% of the overall predictions by the model are correct.
# #### The precision for predicting No Diabetes is 0.78 means that 78% of the time the model predicting a case as No Diabetes is correct.
# #### The precision for predicting Diabetes is 0.84 means that 84% of the time the model predicting a case as Diabetes is correct.
# #### Recall for No Diabetes is 0.84 means that the model can recognize 84% of all the No Diabetes cases.
# #### Recall for Diabetes is 0.78 means that the model can recognize 78% of all the Diabetes cases.
# #### F-1 scores for No diabetes and Diabetes is 0.81, indicating a good balance between precision and recall for both classes & a balanced model's prediction for both case types.
# #### Support of 93 and 101 samples for No Diabetes and Diabetes classes respectively indicates that the dataset is relatively balanced between both classes.
# 
# #### Hence, the model is performing effectively in predicting both cases.

# ### Analysis of Confusion Matrix.
# 
# #### True Positive(TP)- 79, it's the number of times a diabetes case is correctly predicted as diabetes.
# #### True Negative(TN)- 78, it's the number of times a no diabetes case is correctly predicted as no diabetes.
# #### False Positive(FP)- 15, it's the number of times a no diabetes case is incorrectly predicted as diabetes.
# #### False Negative(FN)- 22, it's the number of times a diabetes case is incorrectly predicted no diabetes
# 
# #### The above values from the confusion matrix indicate that-
# #### The model correctly predicts 79 instances of diabetes and 78 instances of no diabetes.
# #### It incorrectly identifies 15 no diabetes cases as diabetes and 22 diabetes cases as no diabetes.
# #### Hence, the model is effective in predicting both cases but still there is some room for improvement and false positive and false negative cases can be reduced further.

# ### ROC-AUC(Receiver Operating Characteristics-Area Under the Curve) Score-
# 
# #### ROC-AUC Score of 0.8104439476205685 indicates that a model has a good ability to distinguish between both the Diabetes and No Diabetes classes. As ROC-AUC Score ranges from 0 to 1, with 1 indicating perfect classification and 0.5 or below indicating no discriminative power.

# #### Hence, overall the stacking model has good performance and is very effective in predicting both classes, with a slight scope for further improvement.

# In[ ]:




