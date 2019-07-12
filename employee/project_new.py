# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 18:58:34 2019

@author: user
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
from datetime import datetime
from sklearn import feature_selection
from sklearn import naive_bayes
from sklearn import tree
from sklearn import utils
from sklearn import ensemble
from sklearn import linear_model
from sklearn import neighbors
import random
import warnings
from scipy import stats
#ignoring future warning
warnings.simplefilter(action='ignore', category=FutureWarning)

#importing dataset
df_p=pd.read_csv("EmployeeAttrition.csv")


#shuffling dataset to get  a randomness
df_p=utils.shuffle(df_p,random_state=42)

#checking unique values of differnet columns
for col in df_p.columns.values:
    print(str(col),df_p[str(col)].unique().shape[0])



#converting Attrition value into 0 and 1
df_p["op"]=(df_p["Attrition"]=="Yes").astype(np.int)
df_p.drop("Attrition",axis=1,inplace=True)  






#dropping 4 columns which is not effecting the results
l_drop=[]
for col in df_p.columns.values:
    if df_p[str(col)].unique().shape[0]==1:
        l_drop.append(str(col))

l_drop.append("EmployeeNumber")
df_p.drop(l_drop,axis=1,inplace=True)








#taking string column names int0o one list
l_str=[]
for col in df_p.columns.values:
    
    if type(df_p[str(col)][0])==str or df_p[str(col)].unique().shape[0]==1:
        l_str.append(str(col))



#taking numeric categorical columns names into one list
l_cat=[]
for col in df_p.drop(l_str,axis=1).columns.values:
    if df_p[str(col)].unique().shape[0]<=10:  
     l_cat.append(str(col))
l_cat.remove("op")




#taking continous columns names into one list
l_con=[]
for col in df_p.drop(l_str,axis=1).columns.values:
    if df_p[str(col)].unique().shape[0]>10:  
     l_con.append(str(col))


#checking for zero or negetive values 
for col in df_p[l_con].columns.values:
    if (df_p[str(col)]<=0).sum()>0:
        print("number of zeros or negative  in columns",str(col),(df_p[str(col)]<=0).sum())
        
        
for col in df_p[l_con].columns.values:
    if (df_p[str(col)]<=0).sum()>0:
        print(df_p[str(col)][df_p[str(col)]<0])
#no negative values
for col in df_p[l_con].columns.values:
    if (df_p[str(col)]<=0).sum()>0:
        print("number of zeros   in columns",str(col),(df_p[str(col)]<=0).sum())
        
        
#further checking on zero values        
for col in df_p[l_con].columns.values:
    if (df_p[str(col)]==0).sum()>0:
        print("zeros in the column",str(col),df_p[str(col)].unique().shape[0],"max",df_p[str(col)].max(),"min",df_p[str(col)].min())
#unique values in these columns containing zeroes
#zeroes are not a anamoly in this case






       

#calculating IQR
def IQR(data):
    upper_quantile=data.quantile(0.75)
    lower_quantile=data.quantile(0.25)
    IQR=upper_quantile-lower_quantile
    outlier1=upper_quantile+1.5*IQR
    outlier2=lower_quantile-1.5*IQR
    return (IQR,outlier1,outlier2)


for col in df_p[l_con].columns.values:
    i,outlier1,outlier2=IQR(df_p[str(col)])
    print("upper_outliers",df_p[df_p[str(col)]>outlier1].shape[0],"column name",str(col),"theroritical_max",outlier1,"max",df_p[str(col)].max())
    print("lower_outliers",df_p[df_p[str(col)]<outlier2].shape[0],"column name",str(col),"theoritical_min",outlier2,"min",df_p[str(col)].min())
#taking outlers into account
#nly monthly income outliers need to be deleted        

#deleting outliers from MonthlyIncome
i,outlier1,outlier2=IQR(df_p["MonthlyIncome"])
df_p.drop(df_p[df_p["MonthlyIncome"]>outlier1].index,inplace=True)



#checking co rerelation
sns.heatmap(df_p[l_con].corr(),annot=True)

df_p.drop([ 'YearsAtCompany',
 'YearsInCurrentRole',
 'YearsSinceLastPromotion'],axis=1,inplace=True)
#deleting corerelated columns     
for i in [ 'YearsAtCompany',
 'YearsInCurrentRole',
 'YearsSinceLastPromotion',
 'YearsWithCurrManager']:
  l_con.remove(i)    
l_cat.append("YearsWithCurrManager")
  
df_p.drop("TotalWorkingYears",axis=1,inplace=True)
l_con.remove("TotalWorkingYears") 

sns.heatmap(df_p[l_con].corr(),annot=True)

#plotting different columns




sns.boxplot(x="op",y="Age",data=df_p)#dependant 
sns.boxplot(x="op",y="DailyRate",data=df_p)#varying but not that much
sns.boxplot(x="op",y="HourlyRate",data=df_p)#almost not varying
sns.boxplot(x="op",y="MonthlyIncome",data=df_p)#no inference
sns.boxplot(x="op",y="MonthlyRate",data=df_p)#not varying
sns.boxplot(x="op",y="PercentSalaryHike",data=df_p)#not varying




sns.violinplot(x="op",y="Age",data=df_p,split=True,inner="quart")#small variation 
sns.violinplot(x="op",y="DailyRate",data=df_p,split=True,inner="quart")#no clear distinction
sns.violinplot(x="op",y="HourlyRate",data=df_p,split=True,inner="quart")#almost not varying
sns.violinplot(x="op",y="MonthlyIncome",data=df_p,split=True,inner="quart")#no clear distinction
sns.violinplot(x="op",y="MonthlyRate",data=df_p,split=True,inner="quart")#no results
sns.violinplot(x="op",y="PercentSalaryHike",data=df_p,split=True,inner="quart")#no distinction

#original plot of Attrition
sns.countplot(x=df_p.op)

sns.countplot(hue="op",x="BusinessTravel",data=df_p)#clear variation
sns.countplot(hue="op",x="Department",data=df_p)#not clear distinction
sns.countplot(hue="op",x="EducationField",data=df_p)#hafazard
sns.countplot(hue="op",x="Gender",data=df_p)#cannot say
sns.countplot(hue="op",x="JobRole",data=df_p)#dependencies may present
sns.countplot(hue="op",x="MaritalStatus",data=df_p)#may be dependent
sns.countplot(hue="op",x="OverTime",data=df_p)#cannot say
sns.countplot(hue="op",x="TrainingTimesLastYear",data=df_p)#no result
sns.countplot(hue="op",x="NumCompaniesWorked",data=df_p)#no 
sns.countplot(hue="op",x="Education",data=df_p)
sns.countplot(hue="op",x="JobInvolvement",data=df_p)
sns.countplot(hue="op",x="JobLevel",data=df_p)
sns.countplot(hue="op",x="JobSatisfaction",data=df_p)
sns.countplot(hue="op",x="PerformanceRating",data=df_p)
sns.countplot(hue="op",x="RelationshipSatisfaction",data=df_p)
sns.countplot(hue="op",x="StockOptionLevel",data=df_p)
sns.countplot(hue="op",x="WorkLifeBalance",data=df_p)


#percentage of yes and no of total datasetis
print("yes",df_p[df_p.op==1].shape[0]/df_p.shape[0],"no",df_p[df_p.op==0].shape[0]/df_p.shape[0])


#checking the dependencies of different class using the percentage of yes or no in every
#category of the categorical variables
cat=l_cat+l_str
for col in df_p[cat].columns.values:
    for  uni in np.sort(df_p[str(col)].unique()):
        temp=df_p["op"][df_p[str(col)]==uni]
        #print(temp)
        print(str(col),uni,"yes",(temp==1).sum()/temp.shape[0],"no",(temp==0).sum()/temp.shape[0])


#one hot encoding categorical columns

df_all_cat=df_p[cat]


df_p=pd.get_dummies(df_p,columns=cat)

df_all_cat.columns.values

df_p.drop(l_con,axis=1).columns.values








#feature scalling over continous data
for col in df_p[l_con]:
    u=df_p[col].max()-df_p[col].min()
    avg=df_p[col].mean()
    df_p[col]=(df_p[col]-u)/avg






















#feature selection using decision tree
x=df_p.drop("op",axis=1)
y=df_p.op
xtrain,xtest,ytrain,ytest=model_selection.train_test_split(x,y,test_size=0.1,random_state=42)

trmodel=tree.DecisionTreeClassifier(max_depth=10)
trmodel.fit(xtrain,ytrain)
f_imp=trmodel.feature_importances_
val=x.columns.values
zzz=pd.DataFrame({"value":val,"fi":f_imp})
pdt=zzz.sort_values(by="fi",ascending=False)   
pdt
#feature selection using extratree classifier

etrmodel=ensemble.ExtraTreesClassifier()
etrmodel.fit(xtrain,ytrain)
fi=etrmodel.feature_importances_
val=x.columns.values
zzz=pd.DataFrame({"value":val,"fi":fi})
petr=zzz.sort_values(by="fi",ascending=False)

#feature selection using random tree classifier
rf=ensemble.RandomForestClassifier(max_depth=10)
rf.fit(xtrain,ytrain)
fi=rf.feature_importances_
val=x.columns.values
zzz=pd.DataFrame({"value":val,"fi":fi})
prf=zzz.sort_values(by="fi",ascending=False)




#using differmet models


def modelstats2(Xtrain,Xtest,ytrain,ytest):
    stats=[]
    modelnames=["LogisticReg","DecisionTree","KNN","NB"]
    models=list()
    models.append(linear_model.LogisticRegression())
    models.append(tree.DecisionTreeClassifier())
    models.append(neighbors.KNeighborsClassifier())
    models.append(naive_bayes.GaussianNB())
    for name,model in zip(modelnames,models):
        if name=="KNN":
            k=[l for l in range(5,17,2)]
            grid={"n_neighbors":k}
            grid_obj = model_selection.GridSearchCV(estimator=model,param_grid=grid,scoring="f1")
            grid_fit =grid_obj.fit(Xtrain,ytrain)
            model = grid_fit.best_estimator_
            model.fit(Xtrain,ytrain)
            name=name+"("+str(grid_fit.best_params_["n_neighbors"])+")"
            print(grid_fit.best_params_)
        else:
            
            model.fit(Xtrain,ytrain)
        trainprediction=model.predict(Xtrain)
        testprediction=model.predict(Xtest)
        scores=list()
        scores.append(name+"-train")
        scores.append(metrics.accuracy_score(ytrain,trainprediction))
        scores.append(metrics.precision_score(ytrain,trainprediction))
        scores.append(metrics.recall_score(ytrain,trainprediction))
        scores.append(metrics.roc_auc_score(ytrain,trainprediction))
        stats.append(scores)
        scores=list()
        scores.append(name+"-test")
        scores.append(metrics.accuracy_score(ytest,testprediction))
        scores.append(metrics.precision_score(ytest,testprediction))
        scores.append(metrics.recall_score(ytest,testprediction))
        scores.append(metrics.roc_auc_score(ytest,testprediction))
        stats.append(scores)
    colnames=["MODELNAME","ACCURACY","PRECISION","RECALL","AUC"]
    return pd.DataFrame(stats,columns=colnames) 

xtr=xtrain[petr["value"][:35]]
xte=xtest[petr["value"][:35]]



modelstats2(xtr,xte,ytrain,ytest)
#using extratree classifier to calculate feature importance and considering total
#35 columns and naive byase








NBmodel=naive_bayes.GaussianNB()
NBmodel.fit(xtr,ytrain)
prediction=NBmodel.predict(xte)
print(metrics.confusion_matrix(ytest,prediction))
print(metrics.recall_score(ytest,prediction))
print(metrics.roc_auc_score(ytest,prediction))

#ROC CURVE
fpr,tpr,thr=metrics.roc_curve(ytest,prediction)
plt.plot(fpr,tpr,color="b")
plt.plot([0,1],[0,1],color="k")











