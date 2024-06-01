import numpy as np
import scipy.io
from numpy import array
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data = scipy.io.loadmat('carpma_veriset.mat')
dataset=data['veri']
dataset = dataset.astype('float32')

a2=dataset.shape[1]
targ=dataset[:,a2-1]
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset[:,0:a2-1])  # 0 ve 1. stütunları yani sadece öznitelikler

train_size = int(len(dataset) * 0.70) #  %70 anlamında
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
targ_train, targ_test = targ[0:train_size], targ[train_size:len(dataset)]
print("Öznitelikler= ",len(train), len(test),"\n")
print("Sonuçlar= ",len(targ_train), len(targ_test))

trainX=train
trainY=targ_train
testX=test
testY=targ_test

model = Sequential()
model.add(Dense(40, input_dim=2, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=5000, verbose=1)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainY = np.reshape(trainY, (trainY.shape[0], 1))
testY = np.reshape(testY, (testY.shape[0], 1))

trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))
#****************************************************************
#soru sorma
soru=array([[-2,3],[0,0],[120000,120000],[16,3]])
soru = scaler.transform(soru)
soruPredict = model.predict(soru)
print('Sorunun cevabı (42-20-144-48):\n',soruPredict)
#****************************************************************

fige2=plt.figure(figsize=(10,5))
plt.plot(trainY,'bo-',label='orijinal eğitim veri')
plt.plot(trainPredict,'r*-',label='train predict')
plt.legend()
plt.show()

fige3=plt.figure(figsize=(10,5))
plt.plot(testY,'bo-',label='orijinal test veri')
plt.plot(testPredict,'r*-',label='test predict')
plt.legend()
plt.show()