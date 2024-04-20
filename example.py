# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt  # 可视化
from matplotlib.font_manager import FontProperties
import matplotlib
font = FontProperties(fname=r"E:\ndlsss\yzl_zyf\static\font\wqy-microhei.ttc", size=14)
import math



import numpy as np

# Generate data for the first segment (0-82)
x1 = np.linspace(0, 82, 82)
y1 = 0.2186 * np.sin(x1 / 82 * np.pi / 2)
z1 = 0.1864 * np.sin(x1 / 82 * np.pi / 2)

# Generate data for the second segment (82-84)
x2 = np.linspace(82, 84, 3)
y2 = [0.2186, 0.2043, 0.19]
z2 = [0.1864, 0.1769, 0.17]

x3 = np.linspace(84, 100, 17)
y3 = np.linspace(0.19, 0, 17)
z3 = np.linspace(0.17, 0, 17)
x = np.concatenate((x1, x2, x3))
y = np.concatenate((y1, y2, y3))
z = np.concatenate((z1, z2, z3))
w = np.sin(x * np.pi / 200)
# data_list = list(zip(x, y))



plt.figure(figsize = (6, 5))  # 可选，设置画布大小

plt.plot(y, color = "blue", label = 'Ip', marker="o", markevery=10)  # linestyle='--'
plt.plot(z, color = "red", label = 'In', marker="^", markevery=10)  # linestyle='--'
plt.plot(w, color = "green", label = 'Ie', marker="s", markevery=10)  # linestyle='--'
# plt.plot(0, 0, 'ro')
# plt.plot(82, 0.2186, 'ro')
# plt.plot(84, 0.19, 'ro')
# plt.plot(100, 0, 'ro')
plt.xlim(0, 100)
plt.ylim(0, 1)
plt.xticks(range(0, 101, 10))
plt.yticks([i/10 for i in range(0, 11)])
plt.xlabel('时间', fontproperties=font)
plt.ylabel('各类人群所占比例者', fontproperties=font)
# plt.title('Training and Test Loss')
plt.legend(prop=font)
# plt.savefig('Training and Test Loss', bbox_inches = 'tight', pad_inches = 0)·
plt.show()


def train(self, inputs, labels, eta, batch_size, epochs=1000):
    dfuzzy_a = [[0 for i in range(self.rule_number)] for j in range(self.fuzzy_set)]
    dfuzzy_b = [[0 for i in range(self.rule_number)] for j in range(self.fuzzy_set)]
    d_subsequence = [[0 for i in range(self.rule_number)] for j in range(self.fuzzy_set + 1)]
    for epoch in range(epochs):
        for k in range(len(inputs)):
            x = inputs[k]
            y = labels[k]
            o, weights, mi_fuzzy, z = self.output(x, train = True)
            weights_sum = sum(weights)
            temp_1 = [[reduce(lambda x, y: x * y, [row[m] for j, row in enumerate(mi_fuzzy) if j != k])
                       for m in range(len(mi_fuzzy[0]))] for k in range(len(mi_fuzzy))]
            for i in range(self.rule_number):
                for k in range(self.fuzzy_set):
                    dfuzzy_a[k][i] += \
                        -(y - o) * \
                        sum([weights[j] * (z[i] - z[j]) if j != i else 0.0 for j in range(self.rule_number)]) / (
                                    weights_sum ** 2) * \
                        temp_1[k][i] * \
                        mi_fuzzy[k][i] * (1 - mi_fuzzy[k][i]) * self.fuzzy_b[k][i]

                    dfuzzy_b[k][i] += \
                        (y - o) * \
                        sum([weights[j] * (z[i] - z[j]) if j != i else 0.0 for j in range(self.rule_number)]) / (
                                    weights_sum ** 2) * \
                        temp_1[k][i] * \
                        mi_fuzzy[k][i] * (1 - mi_fuzzy[k][i]) * (x[k] - self.fuzzy_a[k][i])
                for k in range(self.fuzzy_set + 1):
                    if k == self.fuzzy_set:
                        d_subsequence[k][i] += -(y - o) * weights[i] / weights_sum
                    else:
                        d_subsequence[k][i] += -(y - o) * weights[i] / weights_sum * x[k]
            if (k + 1) % batch_size == 0 or k == len(inputs) - 1:
                for i in range(self.rule_number):
                    for j in range(self.fuzzy_set):
                        self.fuzzy_a[j][i] += -eta[0] * dfuzzy_a[j][i]
                        self.fuzzy_b[j][i] += -eta[0] * dfuzzy_b[j][i]
                    for k in range(self.fuzzy_set + 1):
                        self.subsequence[k][i] += -eta[1] * d_subsequence[k][i]
                    for j in range(self.fuzzy_set):
                        dfuzzy_a[j][i] = 0
                        dfuzzy_b[j][i] = 0
                    for k in range(self.fuzzy_set + 1):
                        d_subsequence[k][i] = 0