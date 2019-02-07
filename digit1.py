import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mping
from scipy.sparse import lil_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sn
import time


# import train dataset
train = pd.read_csv("train.csv")
test_csv = pd.read_csv("test.csv")


#data visualize
for i,j in zip([212,1231,2413,4850,8913,12345,25960,39312],range(8)):
        image = np.array(train.iloc[i,1:])
        image = image.reshape([28,28])
        plt.subplot(2,4,j+1)
        plt.imshow(image,cmap='gray')
        plt.title(i) 
plt.show()
# feature scaling 
train_scaled = train.drop('label',1)
scaler = StandardScaler()
scaler.fit(train_scaled)


start = time.time()
wallclock_start = time.asctime(time.localtime(time.time()))


features = lil_matrix(np.array(train_scaled.iloc[::2,:]), dtype = 'int')
labels = np.array(train.iloc[::2,0]) 


#split trainning set and test setcl
features, test_features, labels, test_labels = train_test_split(features, labels, test_size = 0.1, random_state = 42)
j = 4 # sample in the test.csv
pred_feature = lil_matrix(np.array(test_csv.iloc[j,]),dtype = 'int')


#kfold to split train and test
kf = KFold(n_splits=10)
kf.get_n_splits(features)
best_acc = 0
#find the best k
best_k = 0
#sample number accuracy 
k_value = []
sample_number_acc = []
for k in range(1,10):
        for train_index, valid_index in kf.split(features):
                train_features, valid_features = features[train_index], features[valid_index]
                train_labels, valid_labels = labels[train_index],labels[valid_index]
                #KNN and accuracy
                neigh = KNeighborsClassifier(n_neighbors = k)
                neigh.fit(train_features,train_labels)
                acc = neigh.score(valid_features,valid_labels)
                if acc >= best_acc:
                        best_k = k
                        best_acc = acc
                        best_train_features, best_train_labels = train_features, train_labels
#sample number acc
for k in range(1,50,2):
        total, count = 0, 0
        #KNN and accuracy
        neigh = KNeighborsClassifier(n_neighbors = k)
        neigh.fit(features,labels)
        #sample number acc
        length = len(test_labels)
        sample_number_index = []
        for i in range(length):
            if test_labels[i] == 4:
                sample_number_index.append(i)
                total += 1
        for i in sample_number_index:
            sample_number_pred = neigh.predict(test_features[i])
            sample_number_true = test_labels[i]
            if sample_number_pred == sample_number_true:
                count += 1
        acc1 = count/float(total)
        sample_number_acc.append(acc1)
        k_value.append(k)

#best train_set 
neigh = KNeighborsClassifier(n_neighbors = best_k)
neigh.fit(best_train_features,best_train_labels)
pred = neigh.predict(test_features)
#test set accuracy
acc1 = neigh.score(test_features, test_labels)
print('Accuracy is ', acc1*100, '%')
print('the best k is', best_k)
# confusion matrix 
label_true = test_labels
label_pred = pred
matrix = confusion_matrix(label_true, label_pred)
print(matrix)
'''
matrix_visualize = pd.DataFrame(matrix)
sn.heatmap(matrix_visualize)
'''
#sample number acc
plt.figure()
plt.plot(k_value,sample_number_acc)
plt.xlabel('k value')
plt.ylabel('Accuracy')
plt.title('Accuracy of regonize number 4 with different k value')
plt.show()

end = time.time()
wallclock_end= time.asctime(time.localtime(time.time()))

print('Running time =',end-start,'s')
print(wallclock_start,wallclock_end)
