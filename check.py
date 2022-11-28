import random
import time
import cvxpy as cp
import numpy as np
import os
from matplotlib import pyplot

MAX_ITERATION = 1e+5
matrix_size = 10
s = matrix_size


class Sample:
    def __init__(self):
        self.lamda = 0
        self.Sig1 = np.identity(s)
        self.Sig2 = np.identity(s)
        self.K0 = np.identity(s)
        self.Kstar = np.identity(s)

    def check_t(self, t):
        lhs = np.dot(self.Kstar, np.dot(np.linalg.inv(self.K0 + self.Sig1), (self.K0 - self.Kstar)))
        rhs = np.dot((self.K0 - self.Kstar),
                     (np.identity(matrix_size) - np.dot(np.linalg.inv(self.K0 + self.Sig1), (self.K0 - self.Kstar))))
        B = np.dot(lhs, np.linalg.inv(rhs))

        lhs_t = np.dot(self.Kstar, np.dot(np.linalg.inv(self.K0 + t * self.Sig1), (self.K0 - self.Kstar)))
        rhs_t = np.dot(B, np.dot((self.K0 - self.Kstar),
                                 (np.identity(matrix_size) -
                                  np.dot(np.linalg.inv(self.K0 + t * self.Sig1), (self.K0 - self.Kstar)))))

        flag = True
        abs_ave_diff = 0
        for i in range(matrix_size):
            for j in range(matrix_size):
                abs_ave_diff += abs(lhs_t[i, j] - rhs_t[i, j])

        abs_ave_diff = abs_ave_diff / matrix_size ** 2
        if abs((lhs_t - rhs_t).any()) > 1e-1:
            return [False, abs_ave_diff, np.max(abs(lhs_t - rhs_t))]
        else:
            return [True, abs_ave_diff, np.max(abs(lhs_t - rhs_t))]


diff_set = []
sample_list = []
file = open("lambda=1.4", "r")
line_counter = 0

for i in range(100):
    sample = Sample()
    for j in range(46):
        if j in [0, 1]:
            file.readline()
            continue
        elif 2 <= j <= 11:
            line = file.readline()
            words = line.split()
            for k in range(matrix_size):
                sample.Sig1[j - 2, k] = float(words[k])
        elif 12 <= j <= 23:
            file.readline()
            continue
        elif 24 <= j <= 33:
            line = file.readline()
            words = line.split()
            for k in range(matrix_size):
                sample.K0[j - 24, k] = float(words[k])
        elif j == 34:
            file.readline()
            continue
        elif 35 <= j <= 44:
            line = file.readline()
            words = line.split()
            for k in range(matrix_size):
                sample.Kstar[j - 35, k] = float(words[k])
        elif j == 45:
            file.readline()
            continue
    sample_list.append(sample)
file.close()
print(sample_list[1].Kstar)
t_steps = []
abs_diff_list = []
for step in range(0, 1000):
    realstep = step / 1000 + 0.1
    t_steps.append(realstep)
    tmp = 0
    for i in range(100):
        # print(sample_list[i].check_t(t=realstep))
        tmp += sample_list[i].check_t(t=realstep)[2]
    abs_diff_list.append(tmp / 100)
pyplot.plot(t_steps, abs_diff_list)
pyplot.show()
