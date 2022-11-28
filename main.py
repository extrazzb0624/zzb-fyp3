import random
import time
import cvxpy as cp
import numpy as np
import os

MAX_ITERATION = 1e+5
matrix_size = 10


def f(K, Sig1, Sig2, lamda):
    return np.log(np.linalg.det(K + Sig1)) - lamda * np.log(np.linalg.det(K + Sig2))


def gradient(K, Sig1, Sig2, lamda):
    return (np.linalg.inv(K + Sig1) - lamda * np.linalg.inv(K + Sig2)).T


def steepest_gradient(K, Sig1, Sig2, lamda):
    lamda1, Q = np.linalg.eig(np.linalg.inv(K + Sig1) - lamda * np.linalg.inv(K + Sig2))
    return np.dot(Q, np.dot(np.sign(lamda1) * np.identity(matrix_size), Q.T))


def steepest_gradient_wnos(K, Sig1, Sig2, lamda):
    lamda1, Q = np.linalg.eig(np.linalg.inv(K + Sig1) - lamda * np.linalg.inv(K + Sig2))
    return np.dot(Q, np.dot(lamda1 * np.identity(matrix_size), Q.T))


def normalization(K):
    K = (K + K.T) / 2
    lamda1, Q = np.linalg.eig(K)
    for i in range(matrix_size):
        if lamda1[i] <= 0:
            lamda1[i] = 0
    tmp = np.zeros((matrix_size, matrix_size))
    np.fill_diagonal(tmp, lamda1)
    return np.real(np.dot(Q, np.dot(tmp, Q.T)))


def semidefinite(K, Sig1, Sig2, lamda):
    X = cp.Variable((matrix_size, matrix_size), symmetric=True)
    # The operator >> denotes matrix inequality.
    constraints = [X + K >> 0]
    constraints += [K0 - K >> X]

    g = gradient(K, Sig1, Sig2, lamda).T
    g = -g
    prob = cp.Problem(cp.Minimize(cp.trace(g @ X)),
                      constraints)
    t0 = time.time()
    prob.solve()
    #  print("time for solving problems in this iteration needed is: ", time.time() - t0)
    return X.value


def gain_matrix_modi():
    return np.identity(matrix_size)


def check(K_variable, K_star, gain):
    z = np.dot(np.dot(gain, K_star), np.transpose(gain)) + np.identity(matrix_size)
    lhs = np.dot(np.dot(np.dot(K_star, np.transpose(gain)), z), K_variable - K_star)
    rhs = K_star + np.dot(np.dot(K_star, np.dot(np.transpose(gain), np.linalg.inv(z))),
                          np.dot(K_variable - K_star, np.dot(K_variable - K_star, np.linalg.inv(K_variable))))
    return lhs - rhs


obj_c = []
Final_Result = []
for Lambda in range(14, 20):
    lamb = Lambda / 10 + 0.1
    print(lamb)
    file = open("lambda=%.1f" % lamb, "w+")
    print("file %.1f trial %d" % (lamb, 0))

    for trial in range(100):
        # Random generate initial cases.
        A1 = np.random.normal(0, 1, (matrix_size, matrix_size))
        B1 = np.random.normal(0, 1, (matrix_size, matrix_size))
        C = np.random.normal(0, 1, (2, 2))
        D = np.random.normal(0, 1, (1, 1))
        E = np.random.normal(0, 1, (matrix_size, matrix_size))
        Sig1 = np.dot(A1, A1.T) + np.identity(matrix_size)
        Sig2 = np.dot(B1, B1.T) + np.identity(matrix_size)
        K0 = np.dot(E, E.T) + np.identity(matrix_size) * 5
        K = np.identity(matrix_size)

        # Optimization setting
        iteration = 0
        epsilon = 1e+6
        wrongflag = False
        pre = f(K, Sig1, Sig2, lamb)

        while abs(epsilon) > 1e-3:
            if iteration > MAX_ITERATION:
                wrongflag = True
                break
            iteration += 1
            # print("iteration : ",iteration)
            direction = semidefinite(K, Sig1, Sig2, lamb)
            gamma = min(np.linalg.norm(direction), 0.1)
            # print(gamma)
            K = K + gamma * direction
            K = normalization(K)
            epsilon = f(K, Sig1, Sig2, lamb) - pre
            pre = f(K, Sig1, Sig2, lamb)

        if wrongflag:
            continue

        Kstar = K

        file.write('File %d Trial %d\n ' %(lamb, trial+1))

        file.write('sigma1\n')
        for row in range (0, matrix_size):
            for column in range (0, matrix_size):
                file.write('%.6f ' %Sig1[row][column])
            file.write('\n')
        file.write('sigma2\n')
        for row in range (0, matrix_size):
            for column in range (0, matrix_size):
                file.write('%.6f ' %Sig2[row][column])
            file.write('\n')
        file.write('K0\n')
        for row in range (0, matrix_size):
            for column in range (0, matrix_size):
                file.write('%.6f ' %K0[row][column])
            file.write('\n')
        file.write('Kstar\n')
        for row in range (0, matrix_size):
            for column in range (0, matrix_size):
                file.write('%.6f ' %Kstar[row][column])
            file.write('\n')
        file.write('\n')
        print("file %.1f trial %d" %(lamb, trial+1))
    file.close()



'''for file_number in range(12)
for trial in range(100):
    K = np.identity(matrix_size)
    lamda = np.random.uniform(0,1)+1
    file = open("Opt_")
    obj5 = []
    norm = []
    obj_c = []
    Gamma = []
    gamma = 1
    t0 = time.time()
    pre = f(K, Sig1, Sig2, lamda)
    epsilon = 10
    iteration = 0

    s = 0
    while abs(epsilon) > 1e-3:
        iteration += 1
        # print("iteration : ",iteration)
        obj5.append(pre)
        direction = semidefinite(K, Sig1, Sig2, lamda)

        gamma = min(np.linalg.norm(direction), 0.1)
        Gamma.append(gamma)
        # print(gamma)
        K = K + gamma * direction
        K = normalization(K)
        epsilon = f(K, Sig1, Sig2, lamda) - pre
        pre = f(K, Sig1, Sig2, lamda)
        Final_Result.append(K)

            # print("the value is ",pre)
        #  print(K)
        # K = K + gamma*steepest_gradient(K,Sig1,Sig2,lamda)
        #  K = normalization(K)
        # print("Opt result is", pre, "lambda is ", i, " iteration is ", iteration)
        # for j in Final_Result:
        #    print("check: ", check(j, K, np.identity(matrix_size)))'''
