# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 20:04:32 2024

@author: urmila
"""

#1) Delivery_time -> Predict delivery time using sorting time 
#2) Salary_hike -> Build a prediction model for Salary_hike

#------------------------------------------------------------

#Build a simple linear regression model by performing EDA and do necessary transformations and select the best model using R or Python.
#import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

#import dataset
data=pd.read_csv("/Users/HP/Documents/New folder/delivery_time.csv")
data

#EDA and data visulization
data.info()

sns.distplot(data["delivery Time"])
sns.distplot(data["Sorting Time"])

#renaming columns
dataset=data.rename({"Delivery Time":"delivery_time","Sorting Time":"sorting_time"},axis=1)
dataset

#correlation analysis
dataset.corr()

sns.regplot(x=dataset["sorting_time"],y=dataset["delivery_time"])

#model building
model=smf.ols("delivery_time-sorting_time",data=dataset).fit()
model.summary()

#model testing
#finding coeffcient parameters
model.params

#finding tvalues and pvalues
model.tvalues,model.pvalues

#finding Rsquared and values
model.rsquared,model.rsquared_adj

#model predictions
#manual predication for say sorting time 5
delivery_time=(6.582734)+(1.649020)*(5)
delivery_time

#automatic predication for say sorting time 5,8
new_data=pd.Series([5,8])
new_data

data_pred=pd.DataFrme(new_data,columns=["sorting_time"])
data_pred

model.predict(data_pred)
#_____________________________________________________________________________________________
#2) Salary_hike -> Build a prediction model for Salary_hike
#import libraries
import pandas as pd
#import numpy as pd
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

#import dataset
data1=pd.read_csv("/Users/HP/Downloads/Salary_Data.csv")
data1

#EDA and data visualization
data.info()

sns.distplot(data["YearsExperience"])
sns.distplot(data["Salary"])

#renaming columns
dataset1=data1.rename({"YearsExperience":"Experiance in year"},axis=1)
dataset1

#correlation analysis
dataset1.corr()

sns.regplot(x=dataset1["Experience in year"],y=dataset1["salary"])

#model building
model=smf.ols("Salary-YearsExperience",data1=dataset1).fit()

model.summary()

#model testing
#finding cofficient parameters
model.params

#finding.tvalues and pvalues
model.tvalues,model.pvalues

#finding Rsquared and values
model.rsquared,model.rsquared_adj

#model predictions
#manual predication for say 3years experience
Salary=(25792.200199)+(9449.962321)*(3)
Salary

#automatic predication for say sorting time 5,8
new_data=pd.Series([5,8])
new_data

data_pred=pd.DataFrame(new_data,columns=["YearsExperience"])
data_pred



