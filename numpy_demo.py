#numpy

import numpy as np

arr = np.array([1,2,3,4,5])
print("1d",arr)

arr1 = np.array([[1,2,3],
               [4,5,6]])
print("2d",arr1)


arr2 = np.zeros((2,2))
print("arr2",arr2)

arr3 = np.ones((2,4))
print("aar3",arr3)

arr4 = np.arange(0, 20, 5)
print("aar4",arr4)

arr5 = np.random.rand(2, 2)
print("arr5",arr5)

arr6 = np.array([[1,2,3],[4,5,6]])
print(arr6[1:2])
print("aar6",arr6[-1])
print(arr6.shape)

arr7 = arr6.reshape(3, 2)
print("arr7",arr7)

a = np.array([[1,2], [5,6]])
b = np.array([[4,5], [7,8]])

print("matrix multi",np.dot(a,b))
print("mean", np.mean(a))
print("var", np.var(b))

print("add",a+b)
print("sub",a-b)
print("multi",a*b)
print("divide",a/b)


data1 = np.genfromtxt(r"C:\Users\thaku\Downloads\students_scores.csv", delimiter=",", skip_header=1, usecols=(2, 3, 4))
clm = np.mean(data1, axis=0)
print("clm",clm)
print("data",data1)
print("sum",np.sum(data1))


