import numpy as np

# test shallow copy of transpose
a = np.array([[1.0], [2.0]])
print(a)
(m,n) = a.shape
print(m,n)
b = a.transpose()
print(b)
(m,n) = b.shape
print(m,n)
print(b is a)
print(b.base is a)

# test indexing of array
a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = a[0].transpose()
print(a)
print(a[0].shape)
print(b)
print(b.shape)

# test output
iter = 9
i = 8
a_pairs_changed = 7
print('iter: %d, i: %d, pairs changed: %d' % (iter, i, a_pairs_changed))

# test nonzero()
a = np.array([[1.0, 2.0], [0.0, 4.0], [1.0, 6.0], [0.0, 8.0]])
np.nonzero(a[:, 0])

# test boolean matrix
a = np.array([[1],[2],[3],[4],[5]])
b = (a >= 2)
c = (a <= 4)
indexes = b*c
np. nonzero(indexes)