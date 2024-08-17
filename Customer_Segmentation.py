#!/usr/bin/env python
# coding: utf-8

# #  <font color=blue> ****Predicting Customer Behavior using machine learning algorithms**** </font>

# ## Introduction
# 

# In a rapidly evolving market place , understanding your customer has never been more important. Customer  Segmentation is the process of dividing customers into distinct groups based on shared characterstics such as demographic, behvorial ,geographic and phycological. 
# 
# Customer behavior analysis helps business to better understand their customer needs, preferences and buying pattern, thus helps marketing team to tailor their efforts to reach out customers in the most efficient way for a better customer satisfaction and marketing strategies. The objective of this project is to provide in-depth predictive analysis of customer behavior using machine learning algorithms which plays a vital role in decision making, improve profit rates of business, increase customer satisfaction, and reduce risk by identifying them at the early stage.

# ## Objective 

#  An automobile company has plans to enter new markets with their existing products and after intensive market research, they’ve realized that the behavior of the new market is like their existing market. In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has worked exceptionally well for them. Accordingly, they plan to use the same strategy for the new markets.The objective of the project is to predict the appropriate customer segments(A,B,C,D ) in a new market based on the existing market behavior by leveraging  machine learning techniques thus hepls:
# 
#  -	Personalized marketing:   help the sales team to create highly targeted marketing campaigns focused on the interest and behaviors of specific customer groups, leading to a more satisfying customer experience
#  
#  -	Efficient resource allocation:  business can focus their marketing, sales, and product development effort on segments that respond positively which maximize revenue
#  
#  -	Increase customer retention:  by meeting the specific needs of each segment business can build relationships with customers, increasing loyalty and reducing churn 

# ## Dataset and Data fields :
# 

# The dataset includes information on customers in the existing market , with fields age, gender , martial status, education , profession , work experience , family size, spending score and de-idenified categories. 

# ## Import Necessary libraries

# In[1]:


import pandas as pd # data processing , csv file(e.g. pd.read_s-csv)
import numpy as np  # linear algebra
import seaborn as sns # data visualization
import matplotlib.pyplot as plt # data visualization 
import plotly.express as px      
import plotly.graph_objs as go  # data visualization
from plotly import tools
from plotly.subplots import make_subplots
from  sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.preprocessing  import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from  sklearn.feature_selection import mutual_info_classif
import ipywidgets as widgets
import streamlit as st


plt.figure(figsize=(15,8))
sns.set(rc={'figure.figsize':(12,6)})


# ## Loading and Preprocessing the data

# In[2]:


train = pd.read_csv("https://raw.githubusercontent.com/AbaditEstif/machine_learning/main/Train.csv")
test= pd.read_csv("https://raw.githubusercontent.com/AbaditEstif/machine_learning/main/Test.csv")


# In[3]:


train.shape


# In[4]:


train.duplicated().sum()


# In[5]:


test.shape


# In[6]:


test.duplicated().sum()


# In[7]:


# Concatenate both train and test datasets


data = pd.concat([train,test])


# ### Data understanding 

# My aim is to observe and study the categorical and numerical features of the dataset. by examining the unique value counts. I can understand the distribution and quality of the data and can get insights about the quality of the data,potential issues such as duplicates , imbalances and missing values.

# In[8]:


print("Number of rows:", len(data))
data.head(10)


# In[9]:


data.info()


# In[10]:


data.shape


# In[11]:


# les rename Var_1 column for better understanding 

data= data.rename(columns={"Var_1":"Category"})


# In[12]:


# Checking for misspelled or unsual values in columns


print("Gender:", data['Gender'].unique())
print("Ever_Maried:", data['Ever_Married'].unique())
print("Graduated:", data['Graduated'].unique())
print("Spending_Score:", data['Spending_Score'].unique())
print("Profession:", data['Profession'].unique())
print("Segmentation:", data['Segmentation'].unique())
print("Category:", data['Category'].unique())




# ### Statistical Analysis

# we will use data.describe(), which will give a descriptive overview of the dataset

# In[13]:


# describe Numerical features
   
num_data= data.describe()
num_data.style.background_gradient(cmap='Oranges')



# The above table shows the count,mean standard deviation,min,25%,50%,75% and max values for each column and we can observe that:
# - customer age ranges from 18 to  89 years old with mean of 43.5
# - works experience ranges from 0 to 14 year with mean of 2.64
# - family size ranges from 1 to 9 with  mean of 2.84
# 

# In[14]:


data.duplicated().sum()


# In[15]:


data.columns


# In[16]:


data['Age'].value_counts( dropna= False)


# In[17]:


data['Age'].unique()


# In[18]:


print("Gender:", data['Gender'].value_counts( dropna= False)),'\n',

print("Graduated:", data['Graduated'].value_counts( dropna= False)),'\n',

print("Spending_Score:", data['Spending_Score'].value_counts( dropna= False))


# In[19]:


data['Profession'].value_counts(dropna=False)


# In[20]:


data['Work_Experience'].value_counts(dropna= False)


# In[21]:


data['Family_Size'].value_counts(dropna= False)


# ### Checking and Handling missing values

# In[22]:


# checking missing values

data.isnull().sum()


#  ### Handling missing values

# I will use imputing mechanism to fill the missing value and use mode for categorical columns and median for numerical columns.
# 

# In[23]:


# Replacing missing values in categorical columns with mode and Numerical columns with median

data['Family_Size'].fillna(data['Family_Size'].median(),inplace=True)
data['Work_Experience'].fillna(data['Work_Experience'].median(),inplace=True)

data['Graduated'].fillna(data['Graduated'].mode()[0],inplace=True)
data['Ever_Married'].fillna(data['Ever_Married'].mode()[0],inplace=True)
data['Profession'].fillna(data['Profession'].mode()[0],inplace=True)
data['Category'].fillna(data['Category'].mode()[0],inplace=True)


# In[24]:


data.isnull().sum()


# In[25]:


data.shape


# ###  Checking and handling outliers

# Outliers can be detected using visualizations, implementing mathematical formulas on the dataset or using statistical 
# approach.

# ### Visualizing and removing outliers using Box plot

# It captures the summary of the data with a simple box and whiskers and summarizes using 25th ,50th and 75th percentiles.
# And it can help us to get insights (quartiles, median and outliers ) of the data , potential outliers and understand the
# centeral tendency,

# In[26]:


# Checking outlier for the numerical features

numerical_cols=['Age','Work_Experience','Family_Size']

for column in numerical_cols:
    plt.figure(figsize=(10,6))
    sns.boxplot(x=data[column])
    plt.title(f'{column} Distribution')
    plt.show()




# From the above graph we can observe that work_experience values above 10 are acting as outliers. To address those outliers I will apply IQR(Inter Quartile Range). IQR is the most commonly used and most trusted approach used in the resarch field.
# - IQR= Quartile3-Quartile1, this formula  will provide a measure of the spread of the middle 50% of the data in the work_experience columns .
#  
#  - I am calculating the interquartile rage(IQR) for work experience column,
#  - first computes the first quartile (Q1) and third quartile(Q3) using the midpoint method, then calcualte thr IQR as the difference between Q3 and Q1, providing the middle 50% of work experience.

# In[27]:


Q1 = np.percentile(data['Work_Experience'], 25 , method ='midpoint')

Q3= np.percentile(data['Work_Experience'], 75, method ='midpoint')

IQR= Q3-Q1

print(IQR)


# In[28]:


# let define the upper and lower bound (1.5*IQR)

# Above Uper bound

upper =Q3+1.5*IQR

upper_array =np.array(data['Work_Experience']>= upper)
print("Upper Bound:", upper)
print(upper_array.sum())

# Below lower bound

lower= Q1-1.5*IQR
lower_array = np.array(data['Work_Experience']<= lower)
print("Lower Bound:", lower)
print(lower_array.sum())
 
    


# In[29]:


# let remove the outlier from work experience column

data['Work_Experience']= data['Work_Experience'].apply(lambda x: lower if x<lower else(upper if x>upper else x))


# In[30]:


print(data['Work_Experience'].describe())


# In[31]:


# let examine the modified work experience after removal of the outliers

plt.figure(figsize=(10,6))
sns.boxplot(x= data['Work_Experience'])
plt.title('Box plot for Work Experience')
plt.show()


# In[ ]:





# ## Exploratory Data Analysis(EDA)

# EDA is a crucial step in the data analysis process as it helps studying, exploring, and visualizing information to derive important insights by using statistical tools and visualizations.It aids to find patterns, trends, and relationships between the variables.

# ### Univariate Analysis

# Univare analysis involves looking at the distribution of a single variable.It is an brilliant way to understand a dataset’s range and spread of data. I will use plotly  to create univariate plots quickly.

# #### Calculate Summary Statistics For the Age of the Customer

# In[32]:


data['Age'].mean()


# In[33]:


data['Age'].median()


# In[34]:


data['Age'].std()


# In[35]:


# create frequency table for 'Age'

data['Age'].value_counts()


# In[36]:


fig= px.histogram(data, x="Age", nbins=20, title= "Age Distribution", template="presentation", text_auto=True)
fig.update_layout(bargap=0.1)
fig.show()


# In[ ]:





# #### Percentage distribution of Customer segments

# In[37]:


px.pie(data, names= 'Segmentation', title ="Percentage of each segment",template="presentation")


# In[ ]:





# ####  Gender Distribution  

# In[38]:


gender= data.groupby(["Gender"]).size().rename("count").reset_index()


# In[39]:


gender.head()


# In[40]:


px.bar(gender, x="Gender",y="count",color="Gender",text="count", color_discrete_sequence=["gray","green"], title="Distribution of Gender ",
      template="presentation")


# ####  Distribution of the customer’s professional inclination 

# In[41]:


profession= data.groupby(["Profession"]).size().rename("count").reset_index()


# In[42]:


profession.head()


# In[43]:


px.bar(profession, x="Profession",y="count", color="Profession", text="count", title= " Distribution of Profession ",
      template="presentation")


# In[44]:


#fig= make_subplots(rows= 1,cols=2)
#fig.add_trace(go.bar(x=data['Graduated'].value_counts().index,y=data['Graduated'].value_counts().values),row=1,col=1)
#fig.show()


# In[45]:


px.pie(data_frame= profession, values="count", names="Profession",color="Profession",title=" Distribution of Profession ",
      template="presentation", width= 800,height=600,hole=0.5)


# #### Distribution of Marital Status 

# In[46]:


marital_status= data.groupby(["Ever_Married"]).size().rename("count").reset_index()


# In[47]:


marital_status.head()


# In[48]:


px.bar(marital_status, x= "Ever_Married",y="count",color="Ever_Married", text="count", color_discrete_sequence=["brown","gray"],
      template="presentation")


# In[ ]:





# In[49]:


px.pie(data_frame= marital_status, values="count", names="Ever_Married", color="Ever_Married", title= "Marital Status of the Customer",
      template="presentation")


# #### Distribution  of Educational Level 

# In[50]:


grad= data.groupby(["Graduated"]).size().rename("count").reset_index()


# In[51]:


grad.head()


# In[52]:


px.pie(data_frame= grad,values="count", names="Graduated", color="Graduated", title= " Is the Customer Graduate ", template="presentation") 


# ####   Work Experience in years

# In[53]:


work= data.groupby(["Work_Experience"]).size().rename("count").reset_index()


# In[54]:


work.tail()


# In[55]:


px.bar(work, x="Work_Experience",y="count", text="count", title= "Work Experience in Years", hover_name="Work_Experience",
       color= "Work_Experience",color_discrete_sequence=["orange","red","yellow","brown","gray","blue","purple"],orientation="v",
      template="presentation")


# ####   Spending Score of the Customer

# In[56]:


score= data.groupby(["Spending_Score"]).size().rename("count").reset_index()


# In[57]:


score.head()


# In[58]:


pie_chart= px.pie(data_frame= score, values= 'count', names='Spending_Score',color='Spending_Score',
                  title= 'Spending Score of the Customer',width= 800,height=500,hole=0.6,template="presentation")
pie_chart.show()


# ####  Family members for the customer

# In[59]:


family= data.groupby(["Family_Size"]).size().rename("count").reset_index()


# In[60]:


family.tail()


# In[61]:


bar_chart = px.bar(family, x="Family_Size",y="count", text="count", title= "Family Size of the Customer", hover_name="Family_Size",
       color= "Family_Size",color_discrete_sequence=["orange","red","yellow","brown","gray","blue","purple"],
                template="presentation")
bar_chart.show()


# In[62]:


data.columns


# #### Analysis of Anonymised Category for the Customer

# In[63]:


Anonymized_cat= data.groupby(["Category"]).size().rename("count").reset_index()


# In[64]:


Anonymized_cat.head()


# In[65]:


pie_chart= px.pie(Anonymized_cat, names="Category", values="count",
                  title= "Percentage of Anonymized category for the Customer", template="presentation")
pie_chart.show()


# ### Bivariate Analysis

# Bivariate analysis looks at the relationship between two variables and gives us a better understanding of how the two variables interact.

# #### Customer Segmentation based on Gender

# In[66]:


gender_based= pd.pivot_table(data, values='ID',index=['Gender'],columns=['Segmentation'],aggfunc=np.count_nonzero)


# In[67]:


gender_based


# In[68]:


fig= px.histogram(data, x="Segmentation", color= "Gender", template="presentation", title="Segmentation based on Gender")
                 
fig.update_layout(bargap=0.1)
fig.show()


# ####  Segmentation based on Martial Status

# In[69]:


married_based= pd.pivot_table(data, values='ID',index=['Ever_Married'],columns=['Segmentation'],aggfunc=np.count_nonzero)

married_based


# In[70]:


fig= px.histogram(data, x="Ever_Married",color="Segmentation",barmode="group",template="presentation",
           width=1000,height=600, title= "Segmentation based on Marital Status")
fig.update_layout(bargap=0.2)
fig.show()


# #### Segmentation based on Graduated

# In[71]:


grad_based= pd.pivot_table(data,values='ID',index=['Graduated'],columns=['Segmentation'],aggfunc=np.count_nonzero)

grad_based


# In[72]:


fig= px.histogram(data, x="Graduated",color="Segmentation",barmode="group",template="presentation",
           width=1000,height=600, title= "Customer Segmentation based on Education Level")
fig.update_layout(bargap=0.2)
fig.show()


# #### Segmentation based on Spending score

# In[73]:


spending_based= pd.pivot_table(data, values= 'ID',index=['Spending_Score'],columns=['Segmentation'],aggfunc=np.count_nonzero)

spending_based


# In[74]:


fig= px.bar(data, x="Segmentation",color="Spending_Score", barmode="group",template="presentation",
            title= "Customer Segmentation based on Spending Score")
fig.update_layout(bargap=0.2)
fig.show()


# #### Segmentation based on Profession

# In[75]:


prof_based= pd.pivot_table(data, values= 'ID',index=['Profession'],columns=['Segmentation'],aggfunc=np.count_nonzero)

prof_based


# In[76]:


fig= px.bar(data, x="Segmentation",color="Profession", barmode="group",template="presentation",
            title= " Segmentation based on Profession")
fig.update_layout(bargap=0.2)
fig.show()


# #### Segmentation based on Work Experience

# In[77]:


work_based= pd.DataFrame(data.groupby('Segmentation')['Work_Experience'].mean())
work_based


# In[78]:


fig= px.bar(work_based, x= work_based['Work_Experience'].index, color= "Work_Experience", template="presentation", title="Segmentation based on Work_Experience")
                 
fig.update_layout(bargap=0.1)
fig.show()


# #### Segmentation based o Family Size

# In[79]:


family_based= pd.DataFrame(data.groupby('Segmentation')['Family_Size'].agg(pd.Series.mode))

family_based


# In[80]:


fig= px.bar(family_based, x= family_based['Family_Size'].index, color= "Family_Size", template="presentation", title="Segmentation based on Family Size")
                 
fig.show()


# #### Segmentation based on Category

# In[81]:


var_based= pd.pivot_table(data, values= 'ID',index=['Category'],columns=['Segmentation'],aggfunc=np.count_nonzero)

var_based


# In[82]:


fig= px.bar(data, x="Segmentation",color="Category", barmode="group",template="presentation",
            title= " Segmentation based Category")
fig.update_layout(bargap=0.2)
fig.show()


# #### Segmentation based on Age

# In[83]:


age_based= pd.DataFrame(data.groupby('Segmentation')['Age'].mean())

age_based


# In[84]:


fig= px.bar(age_based, x= age_based['Age'].index, color= "Age", template="presentation", title="Segmentation based on Age")
                 
fig.show()


# In[85]:


sns.pairplot(data =data[['Age','Graduated','Work_Experience','Family_Size','Segmentation']], hue='Segmentation', palette=["Red","Green","yellow",
                                                                                                     "blue"])
plt.show()


# #### Spending score based on Gender

# In[86]:


fig, axes=plt.subplots(1,2)

sns.countplot(data=data, x='Gender',ax= axes[0])

sns.countplot(data=data, x='Gender',hue='Spending_Score',ax=axes[1])
plt.show()


# #### Spending Score Based on Profession

# In[87]:


sns.countplot(data=data, x='Profession',hue='Spending_Score')
plt.show()


# ####  Spending Score based on Family Size

# In[88]:


sns.countplot(data= data, x='Family_Size',hue='Spending_Score')
plt.show()


#  #### Graduated based on Profession

# In[89]:


sns.countplot(data=data,x='Profession', hue='Graduated')
plt.show()


# ## Feature Engineering

# Feature engineering is the process of transforming raw data into features that are suitable for machine learning models. Its main aim is to improve model accuracy by providing more meaningful and relevant information. This includes feature creation and feature transformation:
# 
# -	  Feature creation: is the process of generating new features based on domain know knowledge and helps improve model performance, increase model robustness, improve model interpretability, and increase model flexibility 
# 
# -	 Feature transformation: is the process of transforming the features into suitable representations so that the machine learning can learn effectively from the data. Feature transformation improves model performance, increase model robustness, improve computational efficiency and improve model interpretability. This includes :
# 
# ** Normalization: rescaling the features to have similar range to prevent some feature from dominating others such as 0 and 1 
# 
# ** Scaling: is used to transform numerical variables to have similar scale with a mean 0 and standard deviation 1
# 
# ** Encoding:  this is transforming categorical features into numerical representations. Example label encoding, ordinal encoding and one-hot encoding.
#  
# 
# 
# 

# In[ ]:





# ###  Ordinal Encoding

# In[90]:


df_customer= pd.DataFrame(data)


# In[91]:


df_customer.head()


# In[92]:


# ordinal Endcoding for spending score

encoder =OrdinalEncoder(categories=[['Low', 'Average','High']])
df_customer['Spending_Score'] = encoder.fit_transform(df_customer[['Spending_Score']])
df_customer.head()


# ### Label Encoding

# In[93]:


df_customer['Segmentation'].unique()


# In[94]:


# Label Encoding  for the target varialbe 'Segmentation'

df_customer[['Segmentation']]= df_customer[['Segmentation']].apply(LabelEncoder().fit_transform)

df_customer.head()


# In[95]:


# Label encoder on profession  and Category features

df_customer[['Profession']]= df_customer[['Profession']].apply(LabelEncoder().fit_transform)

df_customer[['Category']]= df_customer[['Category']].apply(LabelEncoder().fit_transform)


df_customer.head()


# In[96]:


# conver categorical variables [ Gender, Graduated , Ever_Married ] using predefined mapping to Numerical values 

df_customer['Gender']= df_customer['Gender'].map({'Male':0, 'Female':1})

df_customer['Graduated']= df_customer['Graduated'].map({'No':0, 'Yes':1})

df_customer['Ever_Married']= df_customer['Ever_Married'].map({'No':0, 'Yes':1})





# In[97]:


df_customer.head()


#  ### Feature selection

# In[ ]:





# In[98]:


# let's check the correlation matrix to visualize how much correltion exist between variables

matrix= df_customer.corr().round(2)
sns.heatmap(matrix,annot=True, cmap='vlag')
plt.show()

    



# In[99]:


x= df_customer.drop('Segmentation', axis=1)
y= df_customer['Segmentation']


# In[100]:


# Let’s identify the columns that significantly influence the segmentation column (the target variable) using 
# mutual information tool from scikit learn package. By checking the values, we will drop columns with little 
# to no influence

importances= mutual_info_classif(x,y)
feature_importances= pd.Series(importances, df_customer.columns[0:len(df_customer.columns)-1])
feature_importances.plot(kind='barh',color='teal')
plt.show()


#  From the above feature important graph, we learn that gender doe does not have any influence on the target variable so let drop it. We need also to drop the ID columns as it is not relevant to our analysis.
#  

# ###  Splitting the dataset into training and testing 

# In[101]:


x= df_customer.drop(['ID','Segmentation', 'Gender'], axis=1)
y= df_customer['Segmentation']


# In[102]:


x.head()


# In[103]:


y.head()


# In[104]:


# split the data into 80% training and 20% testing with random state of 42 to guarantees reproducibility

x_train,x_test, y_train,y_test= train_test_split(x,y,test_size=0.2, random_state=42)


# ### Feature Scaling

# Many machine learning algorithms like logistic regression, KNN, random forest, etc. require data scaling to produce good results.  For this reason, I will use standard scaler to standardize the features.  standard scaler transforms the distribution of each feature to have mean of zero and standard deviation of one. It standardizes features by subtracting the mean value from the feature and dividing the result by the feature standard deviation. This ensures that all features are on the same scale, preventing any single feature from dominating the learning process due to its large magnitude.
# 
# Z=x−μ/σ
# 
# were 
# - x represents the original feature value, 
# - μ is the mean of the feature, 
# - σ is the standard deviation, and 
# - z is the standardized feature value
# 
# 
# 

# In[105]:


# Apply standardization using Standard Scaler 

scaler = StandardScaler()

x_train= scaler.fit_transform(x_train)

x_test= scaler.transform(x_test)


# In[106]:


print(x_train)


# ##  Model Development and Prediction

# ### Logistic Regression

# In[107]:


# Instantiate the model 

logreg= LogisticRegression( max_iter=6000, multi_class='ovr',solver='lbfgs', random_state=42)

# fit the model with data

logreg.fit(x_train,y_train)

y_pred_lr= logreg.predict(x_test)


# #### Model Evaluation

# In[108]:


LRACC= accuracy_score(y_pred_lr,y_test)

print(f" The accuracy score for Logistic regression is {(accuracy_score(y_test,y_pred_lr)*100).round(2)} %")


# Classification report

print(classification_report(y_test,y_pred_lr))



#  Confusion Matrix

cm =confusion_matrix(y_test,y_pred_lr)

print(cm)

# visualizing confusion matrix usig heatmap

ConfusionMatrixDisplay(cm).plot()
plt.tight_layout()
plt.title('confusion Matrix')

plt.show()


# ### K-Nearest Neighbor(KNN)

# K-nearest neighbor or K-NN   is used to solve the classification model problem, by creating an imaginary boundary to classify the data.  Lager k means smother curves of separation resulting less complex model whereas smaller K tends to overfit the data and leads to complex modes.
# 

# In[109]:


# lets find he best value of K uisng cross validation
   
k_values=  [i for i in range(1,50)]  
scores= []
for k in k_values:
   knn= KNeighborsClassifier(n_neighbors=k)
   score= cross_val_score(knn,x,y,cv=5)
   scores.append(np.mean(score))
   
   
sns.lineplot(x= k_values,y= scores, marker ='o')  
plt.xlabel('K Values')
plt.ylabel('Accuracy Score')


# In[ ]:





# In[110]:


# Instantiate the model 

knn= KNeighborsClassifier(n_neighbors= 21)


# fit the model with the data

knn.fit(x_train, y_train)

y_pred_knn = knn.predict(x_test)


# ##### Model Evaluation

# In[111]:


accuracy =accuracy_score(y_test, y_pred_knn)


print(f" The accuracy score for K-NN  is {(accuracy_score(y_test,y_pred_knn)*100).round(2)} %")


# In[112]:


# Classification report

print(classification_report(y_test,y_pred_knn))



#  Confusion Matrix

cm =confusion_matrix(y_test,y_pred_knn)

print(cm)

# visualizing confusion matrix usig heatmap

ConfusionMatrixDisplay(cm).plot()
plt.tight_layout()
plt.title('confusion Matrix')

plt.show()


# ### Support Vector Machine (SVM)

# Support vector machine (SVM) is a powerful machine learning algorithm used for linear or nonlinear classification, regression and for outlier detection.  It is adaptable and efficient in many applications as it can handle high dimensional data and nonlinear relationships.  The main aim of the SVM algorithm is to find optimal hyperplane in an N-dimensional space that can separate the data points in different classes in the feature space.

# In[113]:


# hyper parametre tuning
   
#estimator= SVC()   
#param_grid= {"c":[1,3,5,7,9],"kernel":["linear","polynomial","rbf","sigmoid"]}
#model1= GridSearchCV(estimator, param_grid,cv=5, scoring="accuracy")
#model1.fit(x_train,y_train)
#model1.best_params_
                    


# In[114]:


#Instantiate the model

svm= SVC(C=3, kernel='rbf')
svm.fit(x_train,y_train)

y_pred_svm= svm.predict(x_test)


# #### Model Evaluation

# In[115]:


accuracy =accuracy_score(y_test, y_pred_svm)


print(f" The accuracy score for SVM  is {(accuracy_score(y_test,y_pred_svm)*100).round(2)} %")


# In[116]:


# Classification report

print(classification_report(y_test,y_pred_svm))



#  Confusion Matrix

cm =confusion_matrix(y_test,y_pred_svm)

print(cm)

# visualizing confusion matrix usig heatmap

ConfusionMatrixDisplay(cm).plot()
plt.tight_layout()
plt.title('confusion Matrix')

plt.show()


# ### Decision Tree

# decision trees combine multiple points and weigh degrees of uncertainty to determine the best approach to a complex decision.  It allows us to break down information into multiple variables to arrive at a the best single one to the problem. It also helps companies to create informed opinions that facilitate better decision making. 
# 
# Random forest algorithm differ from decision trees in their ability to form several decisions in order  to reach  a final  decision, it goes a step  further and don’t rely on a single decision like decision tree. 
# 

# #### Hyper parametre Tuning

# In[117]:


estimator= DecisionTreeClassifier()
param_grid= {'criterion': ['gini','entropy']}
model=GridSearchCV(estimator,param_grid,cv=5, scoring='accuracy')
model.fit(x_train,y_train)
model.best_params_


# In[118]:


# Instantiate the model

dt= DecisionTreeClassifier(criterion='gini')


# fit the model with  data 

dt.fit(x_train,y_train)
y_pred_dt= dt.predict(x_test)


# #### Model Evaluation

# In[119]:


accuracy =accuracy_score(y_test, y_pred_dt)

print(f" The accuracy score for Decision tree  is {(accuracy_score(y_test,y_pred_dt)*100).round(2)} %")


# In[120]:


# Classification report

print(classification_report(y_test,y_pred_dt))



#  Confusion Matrix

cm =confusion_matrix(y_test,y_pred_dt)

print(cm)

# visualizing confusion matrix usig heatmap

ConfusionMatrixDisplay(cm).plot()
plt.tight_layout()
plt.title('confusion Matrix')

plt.show()


# ### Random Forest

# #### Hyper parameter Tuning

# In[121]:


estimator= RandomForestClassifier( random_state=42)
param_grid= {'n_estimators': list(range(1,50))}
model=GridSearchCV(estimator,param_grid,cv=5, scoring='accuracy')
model.fit(x_train,y_train)
model.best_params_


# In[122]:


# Instantiate the model

rf= RandomForestClassifier(n_estimators= 41)


# fit the model with  data 

rf.fit(x_train,y_train)
y_pred_rf= rf.predict(x_test)


# #### Model Evaluation

# In[123]:


accuracy =accuracy_score(y_test, y_pred_rf)


print(f" The accuracy score for Random Forest  is {(accuracy_score(y_test,y_pred_rf)*100).round(2)} %\n")




# Classification report

print(classification_report(y_test,y_pred_rf) )



#  Confusion Matrix

cm =confusion_matrix(y_test,y_pred_rf)

print(cm)

# visualizing confusion matrix usig heatmap

ConfusionMatrixDisplay(cm).plot()
plt.tight_layout()
plt.title('confusion Matrix')

plt.show()





# #### Summary

#  According to the performance metrics  support vector machine achieve the highest accuracy which is 47.64% whereas Decision tree achieve the lowest accuracy which is 37.4%
#  
# - 	Logistic regression achieved an accuracy of 44.55% 
# 
# -   K-Nearest Neighbors (KNN) achieved an accuracy of 46%
# - 	Support vector machine (SVM) achieved an accuracy of 47.64%
# - 	Decision tree achieved an accuracy of 37.4%
# - 	Random Forest achieved an accuracy of 41.75%
# 

# In[ ]:




