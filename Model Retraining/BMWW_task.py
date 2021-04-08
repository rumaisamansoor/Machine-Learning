#!/usr/bin/env python
# coding: utf-8

# # Decription - BMWW is to update the model
# 
# #### Approach: In this scenario, i have divided my .csv file into 5 different subsets. One of them i have reserved for validation. I have created different functions to preprocess the data and perform EDA after every file that is read
# 
# 
# #### I would be training on one subset of the dataset using random forest and used different hyper paramters to judge the accuarcy of the model, with the least MSE i have progressed with those particular parameters.
# 
# #### After retraining on every subset of the file, i performed testing on the unforeseen data to check the MSE and i concluded that after every retraining of the model, the accuracy to predict the better results on unforeseen has increased 

# In[128]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


# In[129]:


names=['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility',
       'Item_Type', 'Item_MRP', 'Outlet_Identifier',
       'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type',
       'Outlet_Type', 'Item_Outlet_Sales']


# In[130]:


def read_files(x):   #to read the csv files
    df = pd.read_csv(x,names=names)
    
    
    return df


# In[131]:


def walkthrough(df):   #function to display the general information of the current file
    print("Shape of the current file : ", df.shape,"\n")  
    print("describe function : \n",df.describe(),"\n")  
    uniqueValues = df.nunique(dropna=True)
    print("unique values in each feature : \n",uniqueValues) 


# In[132]:


def null_visual(df):
    
    #visualizing the null values count

    missing_data=df.isnull().sum()
    missing_data[missing_data>0].    sort_values(ascending=True).    plot(kind='barh',figsize=(10,4))
    plt.title('missing value')
    plt.show()  
    missing_data=missing_data[missing_data>0]
    print(missing_data)


# In[133]:


def cat_num(df):
    #lets create a list of categorical feature.
    numeric_data=df.select_dtypes(include = [np.number])
    cat_data=df.select_dtypes(exclude = [np.number])

    print('there are {0} numerical and {1} categorical columns'.    format(numeric_data.shape[1],cat_data.shape[1]))
    
    return numeric_data,cat_data


# In[134]:


def cat_analysis(df):
    
    for col in df:
        print(col)
        print(df[col].value_counts())
        print()


# In[135]:


def na_fill(df):
    #two attributes have missing na, one is numeric, other one is cat
    #filling na with identifier mean 
    df['Item_Weight']=df.groupby('Item_Identifier')['Item_Weight'].apply(lambda x:x.fillna(x.mean()))
    df['Item_Weight'].fillna(0,inplace=True)      #to fill remaining nan values, missed by grouping
    df['Item_Weight'].isnull().sum()
    
    #categorical with mode
    df['Outlet_Size']=df.groupby('Outlet_Type')['Outlet_Size'].apply(lambda x:x.fillna(x.mode()[0]))
    df['Outlet_Size'].isnull().sum()
    
    
    #replacing 0 with mean
    df['Item_Visibility'].replace(0, df['Item_Visibility'].mean(), inplace=True)
    
    return df


# In[136]:


def feature_eng(df):
    
    # "Item_Fat_Content" has similar values with different names so combining them here

    df["Item_Fat_Content"].replace({'LF':'Low Fat', 'reg':'Regular', 'low fat':'Low Fat'}, inplace=True)
    print("after combining similar fat content \n",df["Item_Fat_Content"].value_counts())
    
    
    
    #new feature to store the item type => either food, drink or non consumable
    df['Item_Cat'] = df['Item_Identifier'].apply(lambda x: x[0:2])
    print("\n\nnew variable item_cat value counts \n", df['Item_Cat'].value_counts())
    return df


# In[137]:


def feature_eng2(df):
    #here i am making new category in item fat content, if the item cat == Non consumable then the corresponding
    #fat content will be non edible

    df.loc[df['Item_Cat']=='NC', 'Item_Fat_Content'] = 'Non-Edible'
    
    
    # as outlet establishment year has a large number => impactful on machine training so reducing it
    #2013 is the year of dataset

    df['Outlet_Years'] = 2013 - df['Outlet_Establishment_Year']
    
    return df


# In[138]:


def EDA_1(df):
    #checking distribution of numeric data
    df['Item_Weight'].hist()

def EDA_2(df):
    df['Item_Visibility'].hist(color="red")
    
    
def EDA_3(df):
     sns.distplot(df['Item_MRP']) 
        
def EDA_4(df):
    sns.distplot(df['Item_Outlet_Sales'])    
    
def EDA_5(df):
    sns.countplot(df["Item_Fat_Content"])            #the distribution is not biased 
    
def EDA_6(df):
    data=sns.countplot(df["Item_Type"])
    data.set_xticklabels(data.get_xticklabels(), rotation=90)
    
    
def EDA_7(df):
    #checking the correlation among features.
    corr = df.corr()
    print(sns.heatmap(corr, annot=True))


# In[139]:


def label_encod(df):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    new_df=df  #creating new df for storing the encoded variables

    new_df.drop(columns=['Outlet_Establishment_Year', 'Item_Identifier', 'Outlet_Identifier'], inplace=True)

    new_df['Item_Type'] = le.fit_transform(new_df['Item_Type'])
    
    new_df = pd.get_dummies(new_df, columns=['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Cat'])
   
    return new_df


# ## Reading the dataframe

# ### dataset file 1

# In[140]:


x='bigmartsalesprediction_0.csv'
df0=pd.read_csv(x)
df0.head()       #to read the files


# ## General information regarding the file

# In[141]:


walkthrough(df0)  #general information of the dataset


# In[142]:


null_visual(df0)  #to visualize the null values in the dataset


# In[143]:


numeric_data,cat_data=cat_num(df0)   #to check on numeric and categorical features of the dataset
cat_analysis(cat_data)              #to find the value counts of each categorical feature


# ## Missing value replacement & feature engineering

# In[144]:


#two attributes have missing na, one is numeric, other one is cat
#filling na with identifier mean 

df0=na_fill(df0)


# In[145]:


df0=feature_eng(df0)
df0=feature_eng2(df0)

df0.shape   #2 new features are created


# ## Exploratory Data Analysis

# In[146]:


EDA_1(df0)
#Item_Weight analysis


# In[147]:


EDA_2(df0)  # Item_Visibility analysis
#=>left skewed =>log transformation


# In[148]:


# log transformation
df0['Item_Visibility'] = np.log(1+df0['Item_Visibility'])
df0['Item_Visibility'].hist(color="red")             # somewhat a better distribution


# In[149]:


#Item_MRP analysis
EDA_3(df0)   #4 distinguised categories can be seen here


# In[150]:


#Item_Outlet_Sales analysis

EDA_4(df0)   #left skewed =>normalization using log transformation


# In[151]:


df0['Item_Outlet_Sales'] = np.log(1+df0['Item_Outlet_Sales'])


# In[152]:


#Item_Fat_Content analysis
EDA_5(df0)    #the distribution is not biased 


# In[153]:


#Item_Type analysis
   
EDA_6(df0)


# In[154]:


EDA_7(df0)


#highly negative corr can be seen( need to remove one of them as they are derived from existing)
#it can be seen that sales shows some correlation with item marp


# ## label encoding

# In[155]:


df0_new=label_encod(df0)


# In[156]:


df0_new.head()


# ## Model training on file 1

# In[157]:


from sklearn.model_selection import train_test_split

#X_0 indicates X(independent features) and 0 (0th file)

X_0= df0_new
X_0=X_0.drop(columns=['Item_Outlet_Sales'],axis=1)
Y_0=df0_new['Item_Outlet_Sales']


# In[158]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


def train(model, X, y):
    model.fit(X, y)
    
    # perform cross-validation
    cross_val_score(model, X, y, cv=5)


# In[159]:


from sklearn.ensemble import RandomForestRegressor

# Instantiate model # criterion = 'mse' as the problem is regression 

#model = RandomForestRegressor(n_estimators = 500, criterion = 'mse', max_depth = 5, random_state=192)  #=> mse = 0.25

#model = RandomForestRegressor(n_estimators = 100, criterion = 'mse', max_depth = None, random_state=42)  # => mse = 0.044

model = RandomForestRegressor(n_estimators = 500, criterion = 'mse', max_depth = None, random_state=42)  #=> mse = 0.043

train(model, X_0, Y_0)

# predict the training set
pred = model.predict(X_0)
    
print("Model Report")
print("MSE:",mean_squared_error(Y_0,pred)) 


# In[161]:


importances = model.feature_importances_

features=X_0.columns
indices = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='#3776ab', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[162]:


#training with important features only

X_01=X_0[['Outlet_Type_Grocery Store', 'Item_MRP', 'Item_Visibility','Item_Type','Outlet_Years','Item_Weight' ]]


# In[181]:


train(model, X_01, Y_0)  #accuracy is almost the same, to make it generalized(avoid overfiiting) i will be progressing
                         #with this.
    
pred = model.predict(X_01)
    
print("Model Report")
print("MSE:",mean_squared_error(Y_0,pred))     


# In[182]:


import pickle

# save the model to disk
filename_1 = 'model_1.sav'
pickle.dump(model, open(filename_1, 'wb'))


# ### i have taken 1 (rows=500) file dataset as validation dataset, so checking the accuracy on the unforeseen data after each retraining 
# 

# In[36]:


#dataset validation

x='bigmartsalesprediction_4.csv'
df_train=read_files(x)
df_train.head()


# In[37]:


walkthrough(df_train)


# In[38]:


null_visual(df_train)


# In[39]:


numeric_data,cat_data=cat_num(df_train)   #to check on numeric and categorical features of the dataset
cat_analysis(cat_data)              #to find the value counts of each categorical feature


# In[40]:


#two attributes have missing na, one is numeric, other one is cat
#filling na with identifier mean 

df_train=na_fill(df_train)


# In[41]:


df_train['Item_Visibility'] = np.log(1+df_train['Item_Visibility'])
df_train['Item_Visibility'].hist(color="red") 


# In[42]:


df_train['Item_Outlet_Sales'] = np.log(1+df_train['Item_Outlet_Sales'])


# In[43]:


df_train=feature_eng(df_train)
df_train=feature_eng2(df_train)

df_train.shape   #2 new features are created


# In[44]:


df_train.shape


# In[45]:


df_train_new=label_encod(df_train)


# In[46]:


df_train_new.head(n=3)


# In[47]:


df_train_X=df_train_new[['Outlet_Type_Grocery Store', 'Item_MRP', 'Item_Visibility','Item_Type','Outlet_Years','Item_Weight' ]]
df_train_Y= df_train_new['Item_Outlet_Sales']


# ### predicting on unforeseen

# In[199]:


#predicting on unforeseen

loaded_model_1 = pickle.load(open(filename_1, 'rb'))
pred = loaded_model_1.predict(df_train_X)

phase_1 = mean_squared_error(df_train_Y,pred)
phase_1
#predicting by model version 1


# ### dataset file 2 and its preprocessing

# In[55]:


#dataset 2   =>file2

x='bigmartsalesprediction_1.csv'
df1=read_files(x)
df1.head()


# In[56]:


walkthrough(df1)  #general information of the dataset


# In[57]:


null_visual(df1)  #to visualize the null values in the dataset


# In[58]:


numeric_data,cat_data=cat_num(df1)   #to check on numeric and categorical features of the dataset
cat_analysis(cat_data)              #to find the value counts of each categorical feature


# In[59]:


#two attributes have missing na, one is numeric, other one is cat
#filling na with identifier mean 

df1=na_fill(df1)


# ### dataset file 2 feature engineering & EDA

# In[60]:


df1=feature_eng(df1)
df1=feature_eng2(df1)

df1.shape   #2 new features are created


# In[61]:


EDA_1(df0)
#Item_Weight analysis


# In[62]:


EDA_2(df0)  # Item_Visibility analysis
#=>left skewed =>log transformation


# In[63]:


# log transformation
df0['Item_Visibility'] = np.log(1+df0['Item_Visibility'])
df0['Item_Visibility'].hist(color="red")             # somewhat a better distribution


# In[64]:


#Item_MRP analysis
EDA_3(df0)   #4 distinguised categories can be seen here


# In[65]:


#Item_Outlet_Sales analysis

EDA_4(df1)   #left skewed =>normalization using log transformation


# In[66]:


df1['Item_Outlet_Sales'] = np.log(1+df1['Item_Outlet_Sales'])


# In[67]:


#Item_Fat_Content analysis
EDA_5(df1)    #the distribution is not biased 


# In[68]:


#Item_Type analysis
    
EDA_6(df1)


# In[69]:


df1_new=label_encod(df1)


# In[70]:


df1_new.head(n=3)


# In[71]:


X_1= df1_new
X_1=X_1.drop(columns=['Item_Outlet_Sales'],axis=1)
Y_1=df1_new['Item_Outlet_Sales']


# In[72]:


X_11=X_1[['Outlet_Type_Grocery Store', 'Item_MRP', 'Item_Visibility','Item_Type','Outlet_Years','Item_Weight' ]]
pred = model.predict(X_11)
    
print("Model Report")
print("MSE:",mean_squared_error(Y_1,pred))     #here the model is not trained on unforeseen so the error is 0.31


# ## Model retraining using dataset file 2

# In[184]:


train(model, X_11, Y_1)  #accuracy is almost the same, to make it generalized(avoid overfiiting) i will be progressing
                         #with this.
    
pred = model.predict(X_11)
    
print("Model Report")
print("MSE:",mean_squared_error(Y_1,pred))     #now the huge difference is seen on accuracy as the model is retrained
                                               #on new file


# In[185]:



# save the model to disk
filename_2 = 'model_2.sav'
pickle.dump(model, open(filename_2, 'wb'))


# ### checking the updated model on unforeseen

# In[198]:


#predicting on unforeseen

loaded_model_2 = pickle.load(open(filename_2, 'rb'))
pred = loaded_model_2.predict(df_train_X)

phase_2= mean_squared_error(df_train_Y,pred)
phase_2
#predicting by model version 2


# ## dataset file 3 reading and walking through

# In[82]:


#dataset file 3

x='bigmartsalesprediction_2.csv'
df_2=read_files(x)
df_2.head()


# In[83]:


walkthrough(df_2)


# In[84]:


null_visual(df_2)


# In[85]:


numeric_data,cat_data=cat_num(df_2)   #to check on numeric and categorical features of the dataset
cat_analysis(cat_data)              #to find the value counts of each categorical feature


# In[86]:


#two attributes have missing na, one is numeric, other one is cat
#filling na with identifier mean 

df_2=na_fill(df_2)


# In[87]:


df_2['Item_Visibility'] = np.log(1+df_2['Item_Visibility'])
df_2['Item_Visibility'].hist(color="red") 


# In[90]:


df_2['Item_Outlet_Sales'] = np.log(1+df_2['Item_Outlet_Sales'])


# ### Feature Engineering and EDA

# In[91]:


df_2=feature_eng(df_2)
df_2=feature_eng2(df_2)

df_2.shape   #2 new features are created


# In[92]:


df_2_new=label_encod(df_2)
df_2_new.shape


# In[93]:


df_2_X=df_2_new[['Outlet_Type_Grocery Store', 'Item_MRP', 'Item_Visibility','Item_Type','Outlet_Years','Item_Weight' ]]
df_2_Y= df_2_new['Item_Outlet_Sales']


# ## Model retraining on dataset file 3

# In[188]:


train(model, df_2_X, df_2_Y)  #retraining

pred = model.predict(df_2_X)
    
print("Model Report")
print("MSE:",mean_squared_error(df_2_Y,pred))     


# In[189]:


# save the model to disk  => third retraining

filename_3 = 'model_3.sav'
pickle.dump(model, open(filename_3, 'wb'))


# ### #predicting on unforeseen
# 

# In[197]:


#predicting on unforeseen

loaded_model_3 = pickle.load(open(filename_3, 'rb'))
pred = loaded_model_3.predict(df_train_X)

phase_3= mean_squared_error(df_train_Y,pred)
phase_3
#predicting by model version 3


# ## Dataset file 4 

# In[110]:


#dataset file 4

x='bigmartsalesprediction_3.csv'
df_3=read_files(x)
df_3.head()


# In[111]:


walkthrough(df_3)


# In[112]:


null_visual(df_3)


# In[113]:


numeric_data,cat_data=cat_num(df_3)   #to check on numeric and categorical features of the dataset
cat_analysis(cat_data)              #to find the value counts of each categorical feature


# In[114]:


#two attributes have missing na, one is numeric, other one is cat
#filling na with identifier mean 

df_3=na_fill(df_3)


# In[116]:


df_3['Item_Visibility'] = np.log(1+df_3['Item_Visibility'])
df_3['Item_Outlet_Sales'] = np.log(1+df_3['Item_Outlet_Sales'])


# ## feature engineering  & EDA

# In[117]:


df_3=feature_eng(df_3)
df_3=feature_eng2(df_3)

df_3.shape   #2 new features are created


# In[118]:


df_3_new=label_encod(df_3)
df_3_new.shape


# In[119]:


df_3_new.head()


# In[120]:


df_3_X=df_3_new[['Outlet_Type_Grocery Store', 'Item_MRP', 'Item_Visibility','Item_Type','Outlet_Years','Item_Weight' ]]
df_3_Y= df_3_new['Item_Outlet_Sales']


# ## Model retraining phase 4

# In[191]:


train(model, df_3_X, df_3_Y)  #retraining

pred = model.predict(df_3_X)
    
print("Model Report")
print("MSE:",mean_squared_error(df_3_Y,pred))     


# In[194]:


# save the model to disk  => forth retraining

filename_4 = 'model_4.sav'
pickle.dump(model, open(filename_4, 'wb'))


# In[196]:


#predicting on unforeseen

loaded_model_4 = pickle.load(open(filename_4, 'rb'))
pred = loaded_model_4.predict(df_train_X)

phase_4=mean_squared_error(df_train_Y,pred)
phase_4
#predicting by model version 4


# In[200]:


print("Prediction on unforeseen after first training : ",phase_1)
print("Prediction on unforeseen after second training : ",phase_2)
print("Prediction on unforeseen after third training : ",phase_3)
print("Prediction on unforeseen after forth training : ",phase_4)


# ## Conclusion : After every model retraining, the prediction on unforeseen data file is improved 

# In[ ]:




