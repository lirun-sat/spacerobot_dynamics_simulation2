import numpy as np
import scipy.linalg
from skew_sym import skew_sym
from Cxyz import cxyz
from cross import cross

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






'''

a = np.arange(12).reshape(4, 3)
print('a', a)
print('a.T', a.T)
# a_00 = a[0:2, :]
# print('a_00', a_00)

a_0 = np.concatenate((a[0, :], a[1, :]), axis=0)
print('a_0', a_0)

a1 = np.arange(12).reshape(4, 3)
print('a1', a1)
print('a1.T', a1.T)

a2 = np.concatenate((a.T, a1.T), axis=0)
print('a2', a2)
a3 = np.arange(24).reshape(4, 6)
print('a3', a3)
a4 = a3 @ a2
print('a4', a4)
a2_1 = np.arange(4).reshape(4, 1)
print('a2_1', a2_1)
a2_2 = a2 @ a2_1
print('a2_2', a2_2)


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

'''


a = np.arange(9).reshape(3, 3)
print('a', a)

b = np.array([2, 4, 7])
print('b', b)

B_0 = b.reshape(3, 1)
print('B_0', B_0)

B_1 = np.array([[1],
                [4],
                [7]])
b_01 = cross(B_0, B_1)
print('b_01', b_01)




b_00 = B_0[0, :] * B_0
print('B_0', B_0[0, :])
print('b_00', b_00)

B_1 = B_0.reshape(3, )
print('B_1.shape', B_1.shape)
print('B_1', B_1)

B_2 = cxyz(B_0[0], 0, 0, 1)
print('B_0[0]', B_0[0])
print('B_2.shape', B_2.shape)
print('B_2', B_2)

b_0 = skew_sym(B_1)
print('b_0', b_0)
print('b_0.shape', b_0.shape)

b_1 = np.array([[2], [4], [7]])
print('b_1', b_1)

c = a @ b
print('c', c)
print(c.shape)

c_1 = a @ b_1
print('c_1', c_1)
print(c_1.shape)

a_1 = np.arange(9).reshape(3, 3)

for i in range(3):
    a_1 = scipy.linalg.block_diag(a_1, np.arange(9).reshape(3, 3))
    print('a_1', a_1)





