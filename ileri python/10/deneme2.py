import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

veri=pd.read_excel('Pima.xlsx').values
input=veri[:,0:-1]
target=veri[:,[-1]]
kfold1=KFold(10,shuffle=True,random_state=42)
yy=kfold1.split(target)
acc,pre,rec,f1=list(),list(),list(),list()
predictions_list=np.empty(shape=(0,1))
original_list=np.empty(shape=(0,1))
for tr,tst in yy:
    train_x=input[tr]
    train_y=target[tr]
    test_x=input[tst]
    test_y=target[tst]
    # clf = SVC(kernel='rbf')
    clf=RandomForestClassifier(min_samples_leaf=1)
    clf.fit(train_x,train_y)
    pred=clf.predict(test_x)
    pred=pred[:,np.newaxis]
    acc.append(metrics.accuracy_score(test_y,pred))
    pre.append(metrics.precision_score(test_y, pred))
    rec.append(metrics.recall_score(test_y, pred))
    f1.append(metrics.f1_score(test_y, pred))
    predictions_list=np.concatenate((predictions_list,pred),axis=0)
    original_list=np.concatenate((original_list,test_y),axis=0)

print(metrics.confusion_matrix(original_list,predictions_list))
print('Accuracy=',np.max(acc),'+- ',np.std(acc))
print('Precision=',np.max(pre),'+- ',np.std(pre))
print('Recall=',np.max(rec),'+- ',np.std(rec))
print('F1=',np.max(f1),'+- ',np.std(f1))
plt.boxplot(acc)
plt.show()

