def bubbleSort(array):
    sayac = 0
    for ind in range(len(array)-1):
        if array[ind] > array[ind + 1]:
            array[ind],array[ind + 1] = array[ind + 1],array[ind]
            sayac += 1

    if sayac == 0:
        return array
    else:
        return bubbleSort(array)

dizi = [1,5,2,9,7,6,3,4]
print(bubbleSort(dizi))
