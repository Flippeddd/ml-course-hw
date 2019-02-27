import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from scipy.sparse import lil_matrix
import time 
import math

data = pd.read_csv('data.csv')
# check the data missing and the data features
# print(data.info())

# data visualize
# ax = sns.countplot(data.diagnosis,label = 'Count')
# plt.title('Diagnosis')
# plt.show()

# data preprocessing 
data = data.drop(columns = ['id'])
data[['diagnosis']] = data[['diagnosis']].replace('M', 1)
data[['diagnosis']] = data[['diagnosis']].replace('B', 0)

# scaling 
data_scaled = data.drop(columns = ['diagnosis'])
scaler  = StandardScaler()
scaler.fit(data_scaled)
# print(data.head())
# print(data_scaled.head())

# 10 features with their mean, standard error and worst 
data_mean = data_scaled.iloc[:,:10]
data_se = data_scaled.iloc[:,10:20]
data_worst = data_scaled.iloc[:,20:]

# check the correlation 
def plot_correlation(data,name):
    corr = data.corr()
    plt.figure(figsize = (18,16))
    ax = sns.heatmap(corr, annot = True, fmt = '.1f', linecolor = 'black')
    plt.title('Correlation between ' + name + ' features')
    plt.show()

# plot_correlation(data_mean,'mean')
# plot_correlation(data_se,'standard error')
# plot_correlation(data_worst,'worst')

# when several features are correlated with each other
# we only pick up one of them
data_selected = data_scaled.drop(columns = ['perimeter_mean','area_mean','concavity_mean','concave points_mean',
'perimeter_se','area_se','concavity_se','concave points_worst','perimeter_worst','area_worst','concavity_worst','concave points_worst'])
# print(data_selected.head())
start = time.time()
wallclock_start = time.asctime(time.localtime(time.time()))

# build the features and labels
features = lil_matrix(np.array(data_selected),dtype = 'float')
labels = np.array(data[['diagnosis']])

# split the training set and test set
features, test_features, labels, test_labels = train_test_split(features, labels, test_size = 0.1, random_state = 42)

# KFold cross validation
kf = KFold(n_splits = 10)
kf.get_n_splits(features)
best_acc = 0
for train_index, valid_index in kf.split(features):
    train_features, valid_features = features[train_index], features[valid_index]
    train_labels, valid_labels = labels[train_index], labels[valid_index]
    clf = SVC(gamma='auto')
    clf.fit(train_features, train_labels)
    acc = clf.score(valid_features, valid_labels)
    if acc >= best_acc:
        best_acc = acc
        best_train_features, best_train_labels = train_features, train_labels

# test accuracy
clf = SVC(gamma = 'auto')
clf.fit(best_train_features,best_train_labels)
acc = clf.score(test_features, test_labels)*100
print('The accuracy is %.2f %%' %acc)

# confusion matrix 
label_true = test_labels
label_pred = clf.predict(test_features)
matrix = confusion_matrix(label_true, label_pred)
matrix_visualize = pd.DataFrame(matrix)
sns.heatmap(matrix_visualize, annot = True)
plt.title('Confusion Matrix')
plt.show()

end = time.time()
wallclock_end = time.asctime(time.localtime(time.time()))

# runing time
print('Running time = %.4f s' %(end-start))
print(wallclock_start,wallclock_end)