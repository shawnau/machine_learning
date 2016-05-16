import numpy as np


def load_data(filename):
    dataMatrix = []; labelMatrix=[]
    f = open(filename)
    for line in f.readlines():
        lineToArray = line.strip().split('\t')
        dataMatrix.append([float(lineToArray[0]), float(lineToArray[1])])
        labelMatrix.append([float(lineToArray[2])])
    return np.array(dataMatrix), np.array(labelMatrix)


def random_select_aj(i,m):
    j = i
    while i == j:
        j = int(np.random.uniform(0,m))
    return j


def clip_a(ai, H, L):
    if ai > H:
        return H
    elif ai < L:
        return L
    return ai


def smo_simple(x, y, c, tolerance, maxIter):
    m, n = np.shape(x)
    b = 0
    a = np.zeros((m, 1))
    iter = 0
    while iter < maxIter:
        a_pairs_changed = 0
        for i in range(m):
            f_xi = float((a * y).T.dot(x.dot(x[i].T)) + b)
            Ei = f_xi - float(y[i])

            if ((a[i] < c) and (y[i]*Ei < -tolerance)) or \
               ((a[i] > 0) and (y[i]*Ei > tolerance)):

                j = random_select_aj(i, m)
                a_i_old = a[i].copy()
                a_j_old = a[j].copy()

                f_xj = float((a * y).T.dot(x.dot(x[j].T)) + b)
                Ej = f_xj - float(y[j])

                if y[i] != y[j]:
                    L = max(0, a[j]-a[i])
                    H = min(c, a[j] - a[i] + c)
                else:
                    L = max(0, a[j] + a[i] - c)
                    H = min(c, a[j] + a[i])
                if L == H: print('L == H'); continue

                eta = np.dot(x[i], x[i]) + np.dot(x[j], x[j]) - 2*np.dot(x[i], x[j])
                if eta < 0: print('eta < 0'); continue

                a[j] += y[j]*(Ei - Ej)/eta
                a[j] = clip_a(a[j], H, L)
                if abs(a[j] - a_j_old) < 0.00001:
                    print('j not moving enough, pick another i')
                    continue

                a[i] += y[i]*y[j]*(a_j_old - a[j])
                bi = b - (Ei + y[i] * np.dot(x[i], x[i]) * (a[i] - a_i_old) +
                          y[j] * np.dot(x[j], x[i]) * (a[j] - a_j_old))
                bj = bi + Ei - Ej

                if (a[i] > 0) and (a[i] < c): b = bi
                elif (a[j] > 0) and (a[j] < c): b = bj
                else: b = (bi + bj)/2.0

                a_pairs_changed += 1
                print('iter: %d, i: %d, pairs changed: %d' % (iter, i, a_pairs_changed))
        if a_pairs_changed == 0: iter += 1
        else: iter = 0
        print('iteration: %d' % iter)
    return b, a

