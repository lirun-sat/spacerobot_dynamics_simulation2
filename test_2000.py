import numpy as np



'''
a = np.array([
    [[[1, 2],
      [3, 6]],

     [[6, 2],
      [1, 4]]],


    [[[1, 4],
      [0, 4]],

     [[7, 2],
      [8, 6]]]
])

print(a.shape)
b = np.concatenate((a[0, 0, :, :], a[0, 1, :, :]), axis=1)
c = np.concatenate((a[1, 0, :, :], a[1, 1, :, :]), axis=1)
print(b)
print(c)
d = np.concatenate((b, c), axis=0)
print(d)

b_1 = np.concatenate((a[0, 0, :, :], a[0, 1, :, :]), axis=0)
c_1 = np.concatenate((a[1, 0, :, :], a[1, 1, :, :]), axis=0)
print('b_1.shape', b_1.shape)
print('c_1.shape', c_1.shape)
print(b_1)
print(c_1)
d_1 = np.concatenate((b_1, c_1), axis=0)
print(d_1)
'''

a = np.arange(9).reshape(3, 3)
print(a)
b = np.arange(3)
print(b)

c = a @ b
print(c)

d = np.expand_dims(c, axis=0)
print(d.T)
e = np.array([
    [4],
    [3],
    [7]
])

f = np.concatenate((d.T, e), axis=0)
print(f)
