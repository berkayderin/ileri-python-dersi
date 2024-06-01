import numpy as np
import scipy.io
# from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


veri = scipy.io.loadmat('iris_deneme1.mat')
train_inp=veri['train_inp']
train_tar=veri['train_tar']
test_inp=veri['test_inp']
test_tar=veri['test_tar']

X=np.concatenate((train_inp,test_inp),axis=0)
y=np.concatenate((train_tar,test_tar),axis=0)

kmeans1=KMeans(3)
kmeans1.fit(X)
out_labels=kmeans1.labels_
print('Centers: ',kmeans1.cluster_centers_)
print('Predictions:')
print(kmeans1.predict([[50,30,15,2],[65,36,47,18]]))

plt.scatter(X[:,1],y,marker='o',c='blue')
plt.scatter(X[:,1],out_labels,marker='.',c='red')
plt.show()

acc = accuracy_score(y, out_labels)
silhouette = silhouette_score(y, out_labels)
print("Accuracy: %.2f%%" % (acc * 100))
print("Silhouette score: %.2f" % (silhouette))
