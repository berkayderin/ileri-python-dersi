import matplotlib.pyplot as plt
import numpy as np

# import pandas as pd
# movies_df=pd.read_csv('IMDB-Movie-Data.csv')
# print(movies_df.describe())
# print(movies_df.head(5))
#
# #movies_df.plot(kind='scatter',x="Rating",y='Revenue (Millions)',title='Deneme1')
# # movies_df["Rating"].plot(kind='hist')
# #movies_df["Rating"].plot(kind='box')
# #plt.show()
#
# a1=pd.read_csv('iris.csv',header=None).values
# print(a1)
# #print(a1.head(5))
# a2=pd.read_csv('iris_withheader.csv')
# print('*'*50)
# print(a2.head(5))
#
# a3=pd.read_csv('iris_withheaderandindex.csv',index_col='Id')
# print(a3.head())

x=[-5,-4,-3,-2,-1,0,1,2,3,4,5]
y=[25,16,9,4,1,0,1,4,9,16,25]

x2=np.linspace(-5,5,100)
y2=x2**2
plt.plot(x,y)
plt.plot(x2,y2)
plt.plot(x2,-y2)
# plt.show()
#plt.waitforbuttonpress()
plt.close()

x5=[3,7,19,4,0,44]
plt.plot(x5)
#plt.waitforbuttonpress()
#plt.show()
plt.close()

x6=np.arange(-10,10,1)
y6_1=x6**2
y6_2=-x6**2
plt.plot(x6,y6_1,color='purple',marker='o',markersize=20,linewidth=5)
plt.plot(x6,y6_2,'bo-')
plt.xlabel('x değerleri')
plt.ylabel('y değerleri')
plt.legend(['x^2 eğrisi','-x^2 eğrisi'])
plt.xlim(-5,5)
plt.ylim(-30,30)
plt.title('Deneme')
#plt.savefig('figure1.tiff',dpi=300)
#plt.show()
plt.clf()

sehirler=['Adana','Burdur','İstanbul','Ankara']
sicaklik=[33,12,8,9]
plt.bar(sehirler,sicaklik)
plt.xlabel('Sehirler',fontsize=16)
plt.ylabel('Sıcaklıklar')
plt.show()

