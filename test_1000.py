import numpy as np
import scipy.linalg

'''
a = [[1,2],[3,4],[5,6]]
b = np.asarray(a)
print(b.shape)
print(b)
'''


'''
a_test = 5
for i in range(a_test, -1, -1):
    print(i)

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [11, 12, 13]])
print(a)
print(a.T)
'''


'''
qd = [[], []]

for i in np.arange(0, 2, 0.2):
    qd[0].append(i)
    qd[1].append(i+0.3)

print(qd[0])
print(qd[1])
print(qd[0][3])
print(len(qd))
'''






'''
a = np.array([1, 2, 3, 4, 5])
print(a[4])

dv0 = np.zeros(3)
dw0 = np.zeros(3)
upsilon_base = np.array([dv0, dw0])

print(upsilon_base.shape)
print(upsilon_base)

upsilon_base_tempt1 = upsilon_base.flatten()
print(upsilon_base_tempt1)

HH = np.zeros((12, 12))
print(HH.shape)




print(np.eye(3))
'''




'''
Nl_test = np.arange(36).reshape((3, 3, 2, 2))
print('Nl_test', Nl_test)

Nd_test_1 = np.arange(12).reshape((3, 2, 2))
print('Nd_test_1', Nd_test_1)

Nd_test_2 = np.arange(18).reshape((3, 2, 3))
print('Nd_test_2', Nd_test_2)
'''




'''
test_1 = np.arange(4).reshape((2, 2))
print('test_1', test_1)
test_2 = np.arange(4, 8).reshape((2, 2))
print('test_2', test_2)

test_3 = np.arange(8, 12).reshape((2, 2))
print('test_3', test_3)

test_4 = np.arange(12, 16).reshape((2, 2))
print('test_4', test_4)

test = np.concatenate((test_1,test_2),axis=1)
print('test', test)

test_mul = test_1 @ test_2
print('test_mul', test_mul)
'''




'''
a=np.random.randint(1,9,size=9).reshape((3,3))
print(a)
b=np.random.randint(1,9,size=3)
print(b)

print(b.T)

b=np.expand_dims(b,axis=0)# 注意行向量不能直接转置
print(b)
print(b.T)
'''





'''
d = np.zeros((3, 3))
b = np.zeros((3, 3, 3))
for i in range(3):
    b[i, :, :] = np.arange(1, 10).reshape(3, 3)
    d = scipy.linalg.block_diag(d, b[i, :, :])


# d = scipy.linalg.block_diag(b, c)
# d = scipy.linalg.block_diag(b[i, :, :])

print(d)
print(d.T)
'''




'''
a = np.array([
    [[1, 2, 3],
     [4, 5, 6]],

    [[7, 8, 9],
     [1, 5, 8]]
])

print(a)
print(a.T)
a = np.expand_dims(a, axis=0)
print(a)
print(a.shape)
print(a.T)
'''




'''
b = np.array([
    [[[1, 1],
      [1, 1]],
     [[3, 2],
      [6, 1]],
     [[7, 4],
      [9, 2]]],

    [[[0, 0],
      [0, 0]],
     [[1, 1],
      [1, 1]],
     [[3, 7],
      [8, 4]]],

    [[[0, 0],
      [0, 0]],
     [[0, 0],
      [0, 0]],
     [[1, 1],
      [1, 1]]]
])
# print('b', b)
# print('b.T', b.T)
# print('b.con', b.transpose((1, 0, 2, 3)))
# print((b.transpose((1, 0, 2, 3))).shape)
b_test = b.transpose((1, 0, 2, 3))
print('b_test.shape', b_test.shape)

c = np.array([
    [[3, 5],
     [6, 8]],

    [[9, 5],
     [0, 4]],

    [[1, 4],
     [7, 3]]
])
# print(c)
c = np.expand_dims(c, axis=0)
# print(c.transpose((1, 0, 2, 3)))
c_test = c.transpose((1, 0, 2, 3))
print('c_test.shape', c_test.shape)

#d_test = np.concatenate((c_test, b_test), axis=0)
#print(d_test)
d_test2 = np.concatenate((c_test, b_test), axis=1)
print(d_test2)
print('d_test2.shape', d_test2.shape)

f = np.zeros((4, 2, 2))
f[0, :, :] = np.eye(2)
# print('f.shape', f.shape)
# print(f)
f_1 = np.expand_dims(f, axis=0)
print('f_1.shape', f_1.shape)
# print(f_1)

d_test3 = np.concatenate((f_1, d_test2), axis=0)
print('d_test3', d_test3)
'''





'''
b = np.array([
    [[[1, 0],
      [0, 1]],
     [[3, 2],
      [6, 1]],
     [[7, 4],
      [9, 2]]],

    [[[0, 0],
      [0, 0]],
     [[1, 1],
      [1, 1]],
     [[3, 7],
      [8, 4]]],

    [[[0, 0],
      [0, 0]],
     [[0, 0],
      [0, 0]],
     [[1, 1],
      [1, 1]]]
])

b_test = b.transpose((1, 0, 2, 3))
print('b_test.shape', b_test.shape)
print('b_test', b_test)

c = np.array([
    [[1],
     [0]],
    [[0],
     [1]],
    [[1],
     [1]]
])
print('c.shape', c.shape)
print('c', c)

d = b_test @ c
print('d.shape', d.shape)
print('d:', d)
d_sum = d[1, 0, :, :] + d[1, 1, :, :] + d[1, 2, :, :]
print('d_sum', d_sum)

'''



b = np.array([
    [[[1, 0],
      [0, 1]],
     [[3, 2],
      [6, 1]],
     [[7, 4],
      [9, 2]]],

    [[[0, 0],
      [0, 0]],
     [[1, 1],
      [1, 1]],
     [[3, 7],
      [8, 4]]],

    [[[0, 0],
      [0, 0]],
     [[0, 0],
      [0, 0]],
     [[1, 1],
      [1, 1]]]
])

b_test = b.transpose((1, 0, 2, 3))
print('b_test.shape', b_test.shape)
print('b_test', b_test)

f1 = np.array([
    [[1, 0],
     [0, 1]],
    [[0, 1],
     [1, 1]],
    [[1, 0],
     [1, 1]]
])
print('f1.shape', f1.shape)
print('f1', f1)

d = b_test @ f1
print('d.shape', d.shape)
print('d:', d)

# f_1 = np.expand_dims(f1, axis=0)

# f_1 = f_1.transpose((1, 0, 2, 3))
# print('f_1.shape', f_1.shape)
# print('f_1', f_1)

f2 = np.array([[1, 1],
               [0, 0],
               [0, 1]])
print('f2.shape', f2.shape)
f_2 = np.expand_dims(f2, axis=1)
print('f_2.shape', f_2.shape)
print('f_2:', f_2)
f_3 = f_2.transpose((0, 2, 1))
print('f_3.shape', f_3.shape)
print('f_3:', f_3)












'''
#创建3行4列数组
arr = np.array([[0,1,2,4],[5,6,7,8],[9,10,11,12]])
#用ravel函数展平arr成一维数组arr_b，但arr数组不改变
arr_b = arr.ravel()
# print(arr, arr_b)
arr_b2 = arr_b.reshape((3, 4))
print(arr_b2)
#用flatten展平arr成一维数组arr_c，但arr数组不改变
# arr_c = arr.flatten()
# print(arr, arr_c)
'''








































