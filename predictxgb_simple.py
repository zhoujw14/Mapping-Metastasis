# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 09:07:24 2021

@author: zhouj
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 2018
@author: gangfang

Modified on Fri Sep 13 2019
@author: iannis 
"""
# import necessary libraries
import pandas as pd
import numpy as np 
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score
from time import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, cross_validate, KFold, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt 
import seaborn as sns
from joblib import dump, load
from itertools import cycle
from sklearn.preprocessing import label_binarize
from scipy import interp



#read in the data using pandas
ds=pd.read_csv(r'''C:\JZ\UNC\research\metastasis\relapse analysis\newresult_v7\organ cluster 1\Prediction\predict3.csv''', header=0)

#read dataset info

ds.info(verbose=True)



#load preprocessing functions
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

# Normalize continuous variable(s)
def norm(data, cont_lab):
    for i in cont_lab:
        data[i] = (data[i]-data[i].mean())/data[i].std()
    return data


cat_lab = list(ds.iloc[:,np.r_[1,5,6,7,35,11]])
cont_lab = list(ds.iloc[:,np.r_[3,9,10,12,32,33,36:90,90]])
bin_lab = list(ds.iloc[:,np.r_[4,8,13:31,31,]])

ds1 = norm(ds, cont_lab)
ds1 = onehot(ds, cat_lab, 'notsoftmax')
ds1.info(verbose=True)

# Split input(X) and output(Y) values
X_lab = list(ds1.columns.values)
X_lab.remove('kmeans5')
X_lab.remove('patientID')
X_lab.remove('SUBJID')
X = ds1[X_lab]
Y = ds1['kmeans5']

# Separate data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=169)
print('Train outcome class distribution %s' % Counter(Y_train))
print('Test outcome class distribution %s' % Counter(Y_test))

# perform parameter grid search, using 10-fold cross-validation and F1-score as a metric
# typical values: eta=0.01-0.2, max_depth=3-10, subsample=0.5-1, colsample_bytree: 0.5-1
clf=xgb.XGBClassifier(objective="multi:softprob", random_state=975)
#params = {'max_depth':[3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
#          'min_child_weight': [1, 2, 5], 
#          'learning_rate': [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], 
#          'subsample': [0.25, 0.5, 0.75, 1], 
#          'n_estimators': [50, 100, 200, 500],
#          'colsample_bytree': [0.1, 0.25, 0.5, 0.75, 1]}

params = {'max_depth':[3, 4, 5],
          'min_child_weight': [1, 2, 5], 
          'learning_rate': [0.1], 
          'subsample': [0.5], 
          'n_estimators': [50, 100, 200],
          'colsample_bytree': [0.5]}

params = {'max_depth':[3],
          'min_child_weight': [2], 
          'learning_rate': [0.1], 
          'subsample': [0.5], 
          'n_estimators': [200],
          'colsample_bytree': [0.5]}


cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=999)
xgb=GridSearchCV(clf, param_grid=params, cv=cv, scoring='accuracy', n_jobs=-1)
xgb.fit(X_train, Y_train)

#check top performing n_neighbors value
xgb.best_params_

#check mean score for the top performing value of n_neighbors
xgb.best_score_


# classification report based model performance 
#training
Y_train_pred=xgb.predict(X_train)
print(classification_report (Y_train, Y_train_pred))
#testing
Y_test_pred=xgb.predict(X_test)
print(classification_report (Y_test, Y_test_pred))



#plot ROC AUC curve
accuracy = accuracy_score(Y_test, Y_test_pred) 
print("Accuracy: %.2f%%" % (accuracy * 100.0))

y_test = label_binarize(Y_test, classes=[0, 1, 2,3,4])
y_train = label_binarize(Y_train, classes=[0, 1, 2,3,4])
Y_train_pred_prob=xgb.predict_proba(X_train)
Y_test_pred_prob=xgb.predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = 5
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], Y_test_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), Y_test_pred_prob.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(
    fpr[2],
    tpr[2],
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % roc_auc[2],
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Pattern Predictive Model")
plt.legend(loc="lower right")
plt.show()

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

label1 = list(['Liver First','Lung First','Hetero-Organ','Mono-Organ','Other First'])
# Plot all ROC curves
plt.figure()
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    color="deeppink",
    linestyle=":",
    linewidth=4,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
    color="navy",
    linestyle=":",
    linewidth=4,
)

colors = cycle(["aqua", "darkorange", "cornflowerblue","red",'purple'])
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=lw,
        label="ROC curve of {0} (area = {1:0.2f})".format(label1[i], roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Extension of Pattern Predictive Model")
plt.legend(loc="lower right")
plt.show()
