#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 20:58:28 2018

@author: guowei
"""
#import all pacekages used in this program
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression  
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA as kPCA
from sklearn.model_selection import cross_val_score

#get data
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
                      header=None)

data = data.rename(columns={0: "Class",
                            1: "Alcohol", 2: "Malic_acid",3: "Ash",4:'Alcalinity_of_ash',
                            5: "Magnesium",6:'Total_phenols',7: "Flavanoids",
                            8:'Nonflavanoid_phenols',9: "Proanthocyanins",
                            10:'Color_intensity',11: "Hue",
                            12:'OD280/OD315_of_diluted_wines',13: "Proline",14:'Class'})

#calculate the missing rate of each column
missRate = data.apply(lambda x: (len(x)-x.count())/len(x)*100)
missRate = pd.DataFrame(missRate)
missRate = missRate.rename(columns={0:'Miss_Rate'})

#get basic statistics info of each column
data_stat = data.describe().T

#get types of each column
data_type = pd.DataFrame(data.dtypes)
data_type = data_type.rename(columns={0:'Type'})

#merge the missing rate, basic statistics and types into a dataframe
data_desc = missRate.merge(data_type,how = 'left', left_index = True, right_index = True) \
                .merge(data_stat,how = 'left', left_index = True, right_index = True)

del data_stat, data_type, missRate

#draw a correlation coefficient matrix with a heatmap
cm = data.corr()
plt.subplots(figsize=(9, 9))
plt.title('a correlation coefficient matrix with a heatmap')
hm = sns.heatmap(cm,cbar = True,annot = True,square = True,
                 fmt = '.2f',annot_kws = {'size':10},cmap="YlGnBu")
plt.show()

#draw a boxplot of the features
sc = StandardScaler()
data_std = pd.DataFrame(sc.fit_transform(data))
box = data_std.iloc[:,1:].values
plt.boxplot(box,notch = False, sym = 'rs',vert = True)
plt.title('Features box plot')
#plt.xticks([y for y in range(len(data_std.columns - 1))], data_std.columns)
plt.xlabel('Features')
plt.ylabel('Features value')
plt.show()
del data_std,box

# =============================================================================
# #draw a scatterplot matrix of remaining features
# sns.pairplot(data.iloc[:,0:-1],size = 2.5)
# plt.tight_layout()
# plt.show()
# =============================================================================

#split the dataset into trainning set and test set
X = data.iloc[:,1:].values
y = data.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#build a Logistic Regression model
print('------------------------------------------------------------------------')
print('Logistic Regression Model:')
tuned_parameters = [{'C':np.arange(0.01,1.0,0.01).tolist(),
                     'multi_class':['ovr']}]
clf=GridSearchCV(LogisticRegression(),tuned_parameters,scoring='accuracy',cv=5) 

#find best model params
print('begin searching for best params:')
print()
clf.fit(X_train,y_train)
print('best params found:')
print(clf.best_estimator_)
print()

#fit the model using best params
best_model = clf.best_estimator_
best_model.fit(X_train,y_train)

#print performance indicators
y_train_pred = best_model.predict(X_train)
print('Accruacy of Logistic Regression model(trainning set):')
print(round(metrics.accuracy_score(y_train, y_train_pred),3))

y_test_pred = best_model.predict(X_test)
print('Accruacy of Logistic Regression model(testing set):')
print(round(metrics.accuracy_score(y_test, y_test_pred),3))

print('Confusion_Matrix(trainning set):')
print(confusion_matrix(y_train, y_train_pred))
print('Confusion_Matrix(testing set):')
print(confusion_matrix(y_test, y_test_pred))
print('------------------------------------------------------------------------')

#SVMmodel
print('SVM Model:')
#tuned_parameters = [{'C':np.arange(0.01,1.0,0.05).tolist(),
#                     'gamma':np.arange(0.01,1.0,0.05).tolist()}]
#clf=GridSearchCV(SVC(max_iter = 1000),tuned_parameters,scoring='accuracy',cv=5) 
#
##find best model params
#print('begin searching for best params:')
#print()
#clf.fit(X_train,y_train)
#print('best params found:')
#print(clf.best_estimator_)
#print()
#

#fit the model using best params
#best_model = clf.best_estimator_
#best_model.fit(X_train,y_train)

best_model = svm = SVC(kernel='linear', C=0.95, random_state=1)
best_model.fit(X_train,y_train)
#print performance indicators
y_train_pred = best_model.predict(X_train)
print('Accruacy of SVM model(trainning set):')
print(round(metrics.accuracy_score(y_train, y_train_pred),3))

y_test_pred = best_model.predict(X_test)
print('Accruacy of SVM model(testing set):')
print(round(metrics.accuracy_score(y_test, y_test_pred),3))

print('Confusion_Matrix(trainning set):')
print(confusion_matrix(y_train, y_train_pred))
print('Confusion_Matrix(testing set):')
print(confusion_matrix(y_test, y_test_pred))
print('------------------------------------------------------------------------')



#data standaliztion
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

#PCA model 
#for nn in range(10):
#print(nn)
pca = PCA(n_components=6)
lr = LogisticRegression()
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
print(pca.explained_variance_ratio_.sum())

#PCA_Logistic_Regression_model
print('Logistic Regression Model (PCA):')
tuned_parameters = [{'C':np.arange(0.01,2.0,0.01).tolist(),'multi_class':['ovr']}]
clf = GridSearchCV(LogisticRegression(max_iter = 1000),tuned_parameters,scoring='accuracy',cv=5) 

#find best model params
print('begin searching for best params:')
print()
clf.fit(X_train_pca,y_train)
print('best params found:')
print(clf.best_estimator_)
print()

best_model_lr_pca = clf.best_estimator_
best_model_lr_pca.fit(X_train_pca,y_train)

#best_model_lr_pca = LogisticRegression(C = 0.25)
#best_model_lr_pca.fit(X_train_pca,y_train)





y_train_pred = best_model_lr_pca.predict(X_train_pca)
print('Accuracy of trainning set:')
print(metrics.accuracy_score(y_train, y_train_pred))

y_test_pred = best_model_lr_pca.predict(X_test_pca)
print('Accuracy of testing set:')
print( metrics.accuracy_score(y_test,y_test_pred))

print('confusion_matrix of Trainning set:')
print(confusion_matrix(y_train, y_train_pred))
print('confusion_matrix of Testing set:')
print(confusion_matrix(y_test, y_test_pred))
print('------------------------------------------------------------------------')

#PCA_SVM_model
print('SVM Model (PCA):')
#tuned_parameters = [{'C':np.arange(0.01,1.0,0.05).tolist(),
#                     'gamma':np.arange(0.01,1.0,0.05).tolist()'
#                     'kernel':['linear']}]
#
#clf=GridSearchCV(SVC(max_iter = 1000),tuned_parameters,scoring='accuracy',cv = 5) 
#
#print('begin searching for best params:')
#print()
#clf.fit(X_train_pca,y_train)
#print('best params found:')
#print(clf.best_estimator_)
#print()
#
#best_model_lr_SVM = clf.best_estimator_
#best_model_lr_SVM.fit(X_train_pca,y_train)

best_model_lr_SVM = svm = SVC(kernel='linear', C=0.95, random_state=1)
best_model_lr_SVM.fit(X_train_pca,y_train)
    

y_train_pred = best_model_lr_SVM.predict(X_train_pca)
print('Accuracy of trainning set:')
print(metrics.accuracy_score(y_train, y_train_pred))

y_test_pred = best_model_lr_SVM.predict(X_test_pca)
print('Accuracy of testing set:')
print( metrics.accuracy_score(y_test,y_test_pred))

print('confusion_matrix of Trainning set:')
print(confusion_matrix(y_train, y_train_pred))
print('confusion_matrix of Testing set:')
print(confusion_matrix(y_test, y_test_pred))
print('------------------------------------------------------------------------')


#LDA
print('Logistic Regression Model (LDA):')
lda = LDA(n_components = 3)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)


#LDA_Logistic_Regression_model
tuned_parameters = [{'C':np.arange(0.01,1.0,0.01).tolist(),'multi_class':['ovr']}]
clf=GridSearchCV(LogisticRegression(max_iter = 1000),tuned_parameters,scoring='accuracy',cv=10) 

print('begin searching for best params:')
print()
clf.fit(X_train_lda,y_train)
print('best params found:')
print(clf.best_estimator_)
print()

best_model_lr_lda = clf.best_estimator_

y_train_pred = best_model_lr_lda.predict(X_train_lda)
print('Accuracy of trainning set:')
print(metrics.accuracy_score(y_train, y_train_pred))

y_test_pred = best_model_lr_lda.predict(X_test_lda)
print('Accuracy of testing set:')
print( metrics.accuracy_score(y_test,y_test_pred))

print('confusion_matrix of Trainning set:')
print(confusion_matrix(y_train, y_train_pred))
print('confusion_matrix of Testing set:')
print(confusion_matrix(y_test, y_test_pred))
print('------------------------------------------------------------------------')

#LDA_SVM_model
print('SVM Model (LDA):')
#tuned_parameters = [{'C':np.arange(0.01,1.0,0.05).tolist()}]
#
#clf=GridSearchCV(SVC(max_iter = 1000),tuned_parameters,scoring='accuracy') 
#
#print('begin searching for best params:')
#print()
#clf.fit(X_train_lda,y_train)
#print('best params found:')
#print(clf.best_estimator_)
#print()

best_model_lr_SVM = svm = SVC(kernel='linear', C=0.95, random_state=1)
best_model_lr_SVM.fit(X_train_lda,y_train)

y_train_pred = best_model_lr_SVM.predict(X_train_lda)
print('Accuracy of trainning set:')
print(metrics.accuracy_score(y_train, y_train_pred))

y_test_pred = best_model_lr_SVM.predict(X_test_lda)
print('Accuracy of testing set:')
print( metrics.accuracy_score(y_test,y_test_pred))

print('confusion_matrix of Trainning set:')
print(confusion_matrix(y_train, y_train_pred))
print('confusion_matrix of Testing set:')
print(confusion_matrix(y_test, y_test_pred))
print('------------------------------------------------------------------------')

#kPCA
print('Logistic Regression Model (kPCA):')
kpca = kPCA(n_components = 10, kernel = 'rbf', gamma = 0.01)
X_train_kpca = kpca.fit_transform(X_train_std, y_train)
X_test_kpca = kpca.transform(X_test_std)


#kPCA_Logistic_Regression_model
tuned_parameters = [{'C':np.arange(0.01,2.0,0.01).tolist(),'multi_class':['ovr']}]
clf=GridSearchCV(LogisticRegression(max_iter = 1000),tuned_parameters,scoring='accuracy',cv=10) 

print('begin searching for best params:')
print()
clf.fit(X_train_kpca,y_train)
print('best params found:')
print(clf.best_estimator_)
print()

best_model_lr_kpca = clf.best_estimator_
best_model_lr_kpca.fit(X_train_kpca,y_train)

y_train_pred = best_model_lr_kpca.predict(X_train_kpca)
print('Accuracy of trainning set:')
print(metrics.accuracy_score(y_train, y_train_pred))

y_test_pred = best_model_lr_kpca.predict(X_test_kpca)
print('Accuracy of testing set:')
print( metrics.accuracy_score(y_test,y_test_pred))

print('confusion_matrix of Trainning set:')
print(confusion_matrix(y_train, y_train_pred))
print('confusion_matrix of Testing set:')
print(confusion_matrix(y_test, y_test_pred))
print('------------------------------------------------------------------------')

#kPCA_SVM_model
print('SVM Model (kPCA):')
tuned_parameters = [{'C':np.arange(0.01,1.0,0.05).tolist()}]

clf=GridSearchCV(SVC(max_iter = 1000),tuned_parameters,scoring='accuracy',cv=5) 

print('begin searching for best params:')
print()
clf.fit(X_train_kpca,y_train)
print('best params found:')
print(clf.best_estimator_)
print()

best_model_lr_SVM = clf.best_estimator_
best_model_lr_SVM.fit(X_train_kpca,y_train)
y_train_pred = best_model_lr_SVM.predict(X_train_kpca)
print('Accuracy of trainning set:')
print(metrics.accuracy_score(y_train, y_train_pred))

y_test_pred = best_model_lr_SVM.predict(X_test_kpca)
print('Accuracy of testing set:')
print( metrics.accuracy_score(y_test,y_test_pred))

print('confusion_matrix of Trainning set:')
print(confusion_matrix(y_train, y_train_pred))
print('confusion_matrix of Testing set:')
print(confusion_matrix(y_test, y_test_pred))
print('------------------------------------------------------------------------')


print("My name is Wei Guo")
print("My NetID is: weiguo6")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")