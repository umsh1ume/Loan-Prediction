import math as m
from math import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pylab
import scipy
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

def classification_model(model, data, testdf, predictors, outcome):
    # Fit the model:
    model.fit(data[predictors], data[outcome])

    # Make predictions on training set:
    predictions = model.predict(data[predictors])

    # Print accuracy
    accuracy = metrics.accuracy_score(predictions, data[outcome])
    #print "Accuracy : %s" % "{0:.3%}".format(accuracy)

    # Perform k-fold cross-validation with 5 folds
    kf = KFold(data.shape[0], n_folds=5)
    error = []
    for train, test in kf:
        # Filter training data
        train_predictors = (data[predictors].iloc[train, :])

        # The target we're using to train the algorithm.
        train_target = data[outcome].iloc[train]

        # Training the algorithm using the predictors and target.
        model.fit(train_predictors, train_target)

        # Record error from each cross-validation run
        error.append(model.score(data[predictors].iloc[test, :], data[outcome].iloc[test]))

   # print "Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error))

    # Fit the model again so that it can be refered outside the function:
    model.fit(data[predictors], data[outcome])
    return model.predict(testdf[predictors])

     # Read train file
df=pd.read_csv('train_u6lujuX_CVtuZ9i.csv')
     # Read test file
testdf=pd.read_csv('test_Y3wMUE5_7gLdaTN.csv')

# Let's Know more about data by using data visualisation technique
# This is count of na values
df.isna().sum()



# overview of data
df.describe()


# understand the columns and datatypes
df.dtypes



# Box Plot for understanding the distributions and to observe the outliers.

%matplotlib inline

# Histogram of variable ApplicantIncome

df['ApplicantIncome'].hist()



%matplotlib inline

# Histogram of variable ApplicantIncome

df['Self_Employed'].hist()



# Box Plot for variable ApplicantIncome of training data set

df.boxplot(column='ApplicantIncome')


# Box Plot for variable ApplicantIncome by variable Education of training data set

df.boxplot(column='ApplicantIncome', by = 'Education')


     # Impute missing data in train file with mean values
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df['Self_Employed'].fillna('No', inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(),inplace=True)
df['Credit_History'].fillna(1,inplace=True)
df['Married'].fillna('Yes',inplace=True)
df['Gender'].fillna('Male',inplace=True)
df['Dependents'].fillna('0',inplace=True)
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])

    # Impute missing data in test file with mean values
testdf['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
testdf['Self_Employed'].fillna('No', inplace=True)
testdf['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(),inplace=True)
testdf['Credit_History'].fillna(1,inplace=True)
testdf['Married'].fillna('Yes',inplace=True)
testdf['Gender'].fillna('Male',inplace=True)
testdf['Dependents'].fillna('0',inplace=True)
testdf['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
testdf['TotalIncome_log'] = np.log(df['TotalIncome'])



#print(df.apply(lambda x: sum(x.isnull()), axis=0))

    # Used LabelEncoding to convert all categorical values into numeric
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le=LabelEncoder()
for i in var_mod:
        df[i]=le.fit_transform(df[i])
        if(i!='Loan_Status'):
          testdf[i]=le.fit_transform(testdf[i])

df['CoapplicantIncome'] = df['CoapplicantIncome'].astype(np.int64)
df['LoanAmount']=df['LoanAmount'].astype(np.int64)
df['Loan_Amount_Term']=df['Loan_Amount_Term'].astype(np.int64)
df['Credit_History']=df['Credit_History'].astype(np.int64)
df['TotalIncome']=df['TotalIncome'].astype(np.int64)
df['TotalIncome_log']=df['TotalIncome_log'].astype(np.int64)
df['LoanAmount_log'] = np.log(df['LoanAmount'])

testdf['CoapplicantIncome'] = testdf['CoapplicantIncome'].astype(np.int64)
testdf['LoanAmount']=testdf['LoanAmount'].astype(np.int64)
testdf['Loan_Amount_Term']=testdf['Loan_Amount_Term'].astype(np.int64)
testdf['Credit_History']=testdf['Credit_History'].astype(np.int64)
testdf['TotalIncome']=testdf['TotalIncome'].astype(np.int64)
testdf['TotalIncome_log']=testdf['TotalIncome_log'].astype(np.int64)
testdf['LoanAmount_log'] = np.log(df['LoanAmount'])





#print(df.dtypes)
outcome_var = 'Loan_Status'
model = LogisticRegression()
predictor_var = ['Credit_History','Education','Married','Self_Employed','Property_Area']
p=classification_model(model, df,testdf,predictor_var,outcome_var)
testdf['Loan_Status']=p
df_final=testdf.drop(['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                      'Loan_Amount_Term', 'Credit_History', 'Property_Area','TotalIncome','TotalIncome_log'],axis=1)

df_final['Loan_Status']=df_final['Loan_Status'].map({0:'N', 1:'Y'})
df_final.to_csv('sample_submission.csv', index=False)



# RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
outcome_var = 'Loan_Status'
model = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')
predictor_var = ['Credit_History','Education','Married','Self_Employed','Property_Area']
p=classification_model(model, df,testdf,predictor_var,outcome_var)
testdf['Loan_Status']=p
df_final=testdf.drop(['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                      'Loan_Amount_Term', 'Credit_History', 'Property_Area','TotalIncome','TotalIncome_log'],axis=1)

df_final['Loan_Status']=df_final['Loan_Status'].map({0:'N', 1:'Y'})
df_final.to_csv('sample_submission_RF.csv', index=False)




# DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier

outcome_var = 'Loan_Status'
model = DecisionTreeClassifier()
predictor_var = ['Credit_History','Education','Married','Self_Employed','Property_Area']
p=classification_model(model, df,testdf,predictor_var,outcome_var)
testdf['Loan_Status']=p
df_final=testdf.drop(['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                      'Loan_Amount_Term', 'Credit_History', 'Property_Area','TotalIncome','TotalIncome_log'],axis=1)

df_final['Loan_Status']=df_final['Loan_Status'].map({0:'N', 1:'Y'})
df_final.to_csv('sample_submission_DT.csv', index=False)



#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier


outcome_var = 'Loan_Status'
model = GaussianNB()
predictor_var = ['Credit_History','Education','Married','Self_Employed','Property_Area']
p=classification_model(model, df,testdf,predictor_var,outcome_var)
testdf['Loan_Status']=p
df_final=testdf.drop(['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                      'Loan_Amount_Term', 'Credit_History', 'Property_Area','TotalIncome','TotalIncome_log'],axis=1)

df_final['Loan_Status']=df_final['Loan_Status'].map({0:'N', 1:'Y'})
df_final.to_csv('sample_submission_GNB.csv', index=False)



# gradientboost
from sklearn.ensemble import GradientBoostingClassifier
outcome_var = 'Loan_Status'
model = GradientBoostingClassifier(learning_rate=0.1,n_estimators = 10, random_state = 42)
predictor_var = ['Credit_History','Education','Married','Self_Employed','Property_Area']
p=classification_model(model, df,testdf,predictor_var,outcome_var)
testdf['Loan_Status']=p
df_final=testdf.drop(['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                      'Loan_Amount_Term', 'Credit_History', 'Property_Area','TotalIncome','TotalIncome_log'],axis=1)

df_final['Loan_Status']=df_final['Loan_Status'].map({0:'N', 1:'Y'})
df_final.to_csv('sample_submission_GB.csv', index=False)



# Depending on result we will select model...
# Thanking you





