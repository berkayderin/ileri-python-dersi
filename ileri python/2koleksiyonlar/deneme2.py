liste1=[]
liste2=[1,2,3]
liste3=[1,'iki',3.14]

print(liste1)
print(liste2)
print(liste3)

print('eleman sayısı= ',len(liste3), 'tür')

for i in range(len(liste3)):
    print(liste3[i])

for j in liste3:
    print(j)

print(liste2)
for i in range(len(liste2)):
    liste2[i]=liste2[i]*5
print(liste2)

print(liste3)
for j in liste3:
    j=j*10
print(liste3)

liste4=[1,3,6,8,44,6,8,15,19]
print(liste4)
print("karısık=",liste4[3:7:2])
print('*'*50)

print(liste4[2:5])
print(liste4[5:])
print(liste4[:3])
print(liste4[::-1])
print(liste4[::2])

liste4[3:7:2]=[55,66]
print(liste4)

#liste4[::-1]=[1,2,3,4,5,6,7,8,9]
#print(liste4)

liste4.append(80)
print(liste4)

#liste4.append([1,3])
#print(liste4)
liste4+=[1,3]
print(liste4)

liste4.insert(3,18)
print(liste4)

liste4.remove(3)
print(liste4)

del liste4[1]
print(liste4)

print(liste4.index(1,1))

liste5=[1,2,3,4,5]
liste6=liste5
liste6[2]=33
print(liste5)
print(liste6)
liste7=liste6.copy()
liste7[2]=55
print('*'*50)
print(liste5)
print(liste6)
print(liste7)
print(id(liste5))
print(id(liste6))
print(id(liste7))

print(min(liste7))
print(max(liste7))

liste8=["10A","10B","9A"]
print('minimum=', min(liste8))
print('maksimum=',max(liste8))

print('elemanların toplamı=',sum(liste7))

liste9=[i*3 for i in range(5)]
print(liste9)

liste10=[i*3 for i in range(5) if(i<3)]
print(liste10)
print('*'*50)

demet1=(1,'iki',3.14)
print(type(liste10))
print(type(demet1))
print(demet1)
#demet1[1]=3
print('*'*50)

sozluk1={'a':'aaa','b':2,'c':'ccc'}
print(sozluk1)

print(sozluk1['b'])
sozluk1["d"]=80
sozluk1["b"]='bbb'
print(sozluk1)

del sozluk1['d']
print(sozluk1)

for i in sozluk1:
    print(i)

for i in sozluk1.keys():
    print(i)

for i in sozluk1.values():
    print(i)

print('a' in sozluk1)
print('z' in sozluk1)

sozluk2={'d':'ddd','b':3.14}

print(sozluk1)
sozluk1.update(sozluk2)
print(sozluk1)
