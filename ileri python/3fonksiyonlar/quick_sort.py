def quick_sort(array):
    if len(array) <=1:
        return array
    else:
        pivot = array[len(array) // 2]
        left = [i for i in array if i < pivot]
        middle = [i for i in array if i == pivot]
        right = [i for i in array if i > pivot]
        return quick_sort(left) + middle + quick_sort(right)

arr = [1,5,2,9,7,6,3,4]
print(quick_sort(arr))

