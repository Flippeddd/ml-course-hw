import numpy as np
import pandas as pd
import time
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.sparse import lil_matrix


data = pd.read_csv('diabetes.csv')
# data visualize
#features
feature_names = data.columns[:-1]
plt.subplots(figsize = (18,15))
for (i,j) in zip(feature_names,range(8)):
    plt.subplot(2,4,j+1)
    data[i].hist(bins = 10, edgecolor = 'black')
    plt.title(i)
plt.show()
#outcome 
'''
data['Outcome'].hist(bins = 5, edgecolor = 'black')
plt.title('Outcome')
plt.show()
'''

# Missing data (Due to the large number of zero in this dataset)
print(data.isnull().sum())
data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
print(data.isnull().sum())

# There are some zero value in these data which doesnt make any sense. 
# drop the Insulin part and fill the other data missing with the median.
data = data.drop(columns = ['Insulin'])
print(data.head())
data[['Glucose','BloodPressure','SkinThickness','BMI']] = data[['Glucose','BloodPressure','SkinThickness','BMI']].fillna(data[['Glucose','BloodPressure','SkinThickness','BMI']].median())
print(data.isnull().sum())


# data visualize after pre-processing
feature_names = data.columns[:-1]
plt.subplots(figsize = (18,15))
for (i,j) in zip(feature_names,range(7)):
    plt.subplot(2,4,j+1)
    data[i].hist(bins = 10, edgecolor = 'black')
    plt.title(i)
plt.show()

start = time.time()
#feature scaling
data_scaled = data.drop(['Outcome'],axis = 1)
scaler = StandardScaler()
scaler.fit(data_scaled)

#build features and labels
features = lil_matrix(np.array(data_scaled.iloc[1:,:]),dtype = 'float')
labels = np.array(data.iloc[1:,-1])

#split the training set and test set
features, test_features, labels, test_labels = train_test_split(features, labels, test_size = 0.1, random_state = 42)

#kfold cross validation
kf = KFold(n_splits= 10)
kf.get_n_splits(features)
best_acc,best_k = 0,0
l = len(data['Outcome'])
k_range = int(math.sqrt(l))
for k in range(1,k_range+1):
    for train_index, valid_index in kf.split(features):
        train_features, valid_features = features[train_index], features[valid_index]
        train_lables, valid_labels = labels[train_index], labels[valid_index]
        #KNN
        neigh = KNeighborsClassifier(n_neighbors = k)
        neigh.fit(train_features,train_lables)
        acc = neigh.score(valid_features, valid_labels)
        if acc >= best_acc:
            best_k = k
            best_acc = acc
            best_train_features, best_train_labels = train_features, train_lables

#test accuracy
neigh.fit(best_train_features, best_train_labels)
acc1 = neigh.score(test_features, test_labels)
print('The best accuracy is ', round(acc1*100), '% when k is ', best_k)

'''
#false positive: pred = 1 outcome = 0 
#false negative: pred = 0 outcome = 1
#true postive: pred = 1 outcome = 1
#true negative: pred = 0 outcome = 0
'''
#False positive, false negative, true positive, true negative
pred = neigh.predict(test_features)
l = len(pred)
print(l)
fp_count, fn_count, tp_count, tn_count = 0, 0, 0, 0
for i in range(0,l):
    if pred[i] == 1 and test_labels[i] == 0:
        fp_count += 1
    if pred[i] == 0 and test_labels[i] == 1:
        fn_count += 1
    if pred[i] == 1 and test_labels[i] == 1:
        tp_count += 1
    if pred[i] == 0 and test_labels[i] == 0:
        tn_count += 1
print(fp_count, fn_count, tp_count, tn_count)
l = float(l)
print('the percentage of false positive:', round(fp_count/l*100,2), '%')
print('the percentage of false negative:', round(fn_count/l*100,2), '%')
print('the percentage of true positive:', round(tp_count/l*100,2), '%')
print('the percentage of true negative:', round(tn_count/l*100,2),'%')

#running time
end = time.time()
print('Running time =',end-start,'s')