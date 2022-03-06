# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 10:10:22 2021

@author: zhouj
"""

# import csv data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
from sklearn.metrics import silhouette_score

organdf = pd.read_csv(r'''C:\JZ\UNC\research\metastasis\relapse analysis\newresult_v7\organdf.csv''', header=0)



# One-hot encode categorical variable(s)
def onehot(data, cat_lab, statement):
    ohe = OneHotEncoder(handle_unknown='ignore')
    for i in cat_lab:
        df = pd.DataFrame(ohe.fit_transform(data[[i]]).toarray())
        if statement == 'softmax':
            df = pd.DataFrame(softmax(df, axis=1), index=df.index, columns=df.columns)
        elif statement != softmax:
            print(statement)
        df.columns = [i + str(col) for col in df.columns]
        data = pd.concat([data,df], axis=1)
        data.drop(columns=i, inplace=True)
    return data

categorical = list(organdf.loc[:, organdf.columns.str.startswith('X')].columns)
ds1 = onehot(organdf, categorical, 'softmax') # Preprocessing (one-hot only) #why not prob?


# Split input(X) 
predictors = list(ds1.columns.values)
predictors.remove('patientID')
X = ds1[predictors]

import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans

#plot optimal K basic
Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
print(Sum_of_squared_distances)

#k=3
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
label = kmeans.fit_predict(X)
print(label)


#k=4
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
label = kmeans.fit_predict(X)
print(label)

#k=5
kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
label = kmeans.fit_predict(X)
print(label)

#k=6
kmeans = KMeans(n_clusters=6, random_state=0).fit(X)
label = kmeans.fit_predict(X)
print(label)



#Silhouette score
sil = []
kmax = 10

# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(2, kmax+1):
  kmeans = KMeans(n_clusters = k).fit(X)
  labels = kmeans.labels_
  sil.append(silhouette_score(X, labels, metric = 'euclidean'))


K = range(2, kmax+1)
plt.plot(K, sil, 'bx-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method For Optimal k')
plt.show()
print(sil)







