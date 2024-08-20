# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 14:54:57 2024

@author: Dell
"""
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#load dataset
iris=load_iris()
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['species']=pd.Categorical.from_codes(iris.target,iris.target_names)
#describe dataset
print(df.describe())
#scatterplot
plt.figure(figsize=(8,6))
sns.scatterplot(x='petal length (cm)',y='petal width (cm)',hue='species',data=df,palette='Set1')
plt.title('Scatter plot of petal length vs petal width')
plt.xlabel('petal length (in cm)')
plt.ylabel('petal wdth (in cm)')
plt.show()

