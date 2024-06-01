def fibonacci_recursive(n):
   if n <= 1:
       return n
   else:
       return(fibonacci_recursive(n-1) + fibonacci_recursive(n-2))

terim = 10
dizi=[]
for i in range(terim):
    dizi.append(fibonacci_recursive(i))

print(dizi)
