# -*- coding: utf-8 -*-
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce  # 用于一句话代码完成列表中元素的累乘

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决图片中文无法显示
plt.rcParams['axes.unicode_minus'] = False  # 解决图片中文无法显示


class ANFIS:
    def __init__(self, fuzzy_set, rule_number):
        self.fuzzy_set = fuzzy_set
        # ☆模糊集下隶属函数的个数可以通过优化算法优化
        self.rule_number = rule_number  # 模糊集下隶属函数的个数，标准ANFIS是两个模糊集，就是x对应A下有A1 A2，y对应B下有B1 B2，类似BP每层的神经元个数
        random.seed(69)
        # fuzzy_a,fuzzy_b模糊规则前件的参数集合，控制第一层的计算sigmoid的参数，并非第一层方框里的东西，而是方框内所需要的对应参数，比如A_1的sigmoid输出需要A_a[0]和A_b[0]
        self.fuzzy_a = [[random.random() * 0.5 - 1 for i in range(rule_number)] for j in range(self.fuzzy_set)]  # -0.5 ~ 0.5 i代表着第A、B个模糊集
        self.fuzzy_b = [[random.random() * 0.5 - 1 for i in range(rule_number)] for j in range(self.fuzzy_set)]  # -0.5 ~ 0.5
        # subsequence[0],subsequence[ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15]...模糊规则后件的参数集合，控制第四层的计算px+qy+r的参数
        self.subsequence = [[random.random() * 0.5 - 1 for i in range(rule_number)] for j in range(self.fuzzy_set+1)]  # -0.5 ~ 0.5
        self.MSEs = []
        self.RMSEs = []
        self.MAEs = []
        self.R_squareds = []
        self.Correlations = []
        self.Biass = []

    def sigmoid(self, x, a, b):
        return 1. / (1 + math.exp(b * (x - a)))

    def output(self, input, train=False):
        mi_fuzzy = [[] for j in range(self.fuzzy_set)]  # 存放所有模糊集下规则的隶属度
        antecedent = []  # 记录规则适应度
        z = []  # 记录第四层的输出
        result_layer4 = [sum([self.subsequence[i][j] * input[i] for i in range(len(input))]) + self.subsequence[-1][j] for j in range(self.rule_number)]
        for i in range(self.rule_number):
            for j in range(self.fuzzy_set):
                # A_a[i]和A_b[i]是需要优化的参数，此处用的sigmoid，标准ANFIS用的高斯函数，俩参数分别代表高斯函数的中心和宽度
                mi_fuzzy[j].append(self.sigmoid(input[j], self.fuzzy_a[j][i], self.fuzzy_b[j][i]))  # 第一层的第j个模糊集结果，即隶属度，指定给定的x满足A_1,A_2,...A_rule_number的程度
            antecedent.append(reduce(lambda x, y: x * y, [sublist[i] for sublist in mi_fuzzy]))  # 第二层的 $\Pi$ 结果，即规则的触发强度，记录w_1,w_2,...w_rule_number的结果
            # 有第三层 的归一化 在第五层的结果里，分母放了权重的总和
            z.append(result_layer4[i])  # 第四层的结果

        antecedent_sum = sum(antecedent)  # 权重之和
        o = 0  # 第五层的结果
        for i in range(self.rule_number):
            o += antecedent[i] * z[i]
        o /= antecedent_sum  # 第五层的总输出o

        if train:
            return o, antecedent, mi_fuzzy, z  # 第五层总输出，第三层权重即规则适应度列表，第一层模糊集A下规则的隶属度，第一层模糊集B下规则的隶属度，第四层的结果
        return o

    def train(self, inputs, labels, eta, batch_size, epochs=1000, print_error=100):
        dfuzzy_a = [[0 for i in range(self.rule_number)] for j in range(self.fuzzy_set)]  # （*）记录当次迭代的前件参数
        dfuzzy_b = [[0 for i in range(self.rule_number)] for j in range(self.fuzzy_set)]
        d_subsequence = [[0 for i in range(self.rule_number)] for j in range(self.fuzzy_set + 1)]  # （*）记录当次迭代的后件参数

        for epoch in range(epochs):  # 整个ANFIS

            # print("第%d次迭代" % epoch)
            for k in range(len(inputs)):
                x = inputs[k]
                y = labels[k]
                # x[0]是坐标的x，y[0]是坐标的y
                # 第五层总输出，第三层权重即规则适应度列表，第一层模糊集A下规则的隶属度，第一层模糊集B下规则的隶属度，第四层的结果
                o, weights, mi_fuzzy, z = self.output(x, train = True)
                weights_sum = sum(weights)

                # dw/dμ 生成一个和 mi_fuzzy 结构相同的列表temp_1，其中列表 a 的子列表的每一个元素都是 mi_fuzzy 中除了对应子列表外的其他子列表的对应位置元素的累乘。
                temp_1 = [[reduce(lambda x, y: x * y, [row[m] for j, row in enumerate(mi_fuzzy) if j != k])
                           for m in range(len(mi_fuzzy[0]))] for k in range(len(mi_fuzzy))]
                for i in range(self.rule_number):
                    # GD 梯度下降算法
                    # 反向传播修正前件参数 需要的偏导数，详细推导过程见notability
                    for k in range(self.fuzzy_set):  # j代表着第A、B个模糊集
                        dfuzzy_a[k][i] += \
                            -(y - o) * \
                            sum([weights[j] * (z[i] - z[j]) if j != i else 0.0 for j in range(self.rule_number)]) / (weights_sum ** 2) * \
                            temp_1[k][i] * \
                            mi_fuzzy[k][i] * (1 - mi_fuzzy[k][i]) * self.fuzzy_b[k][i]

                        dfuzzy_b[k][i] += \
                            (y - o) * \
                            sum([weights[j] * (z[i] - z[j]) if j != i else 0.0 for j in range(self.rule_number)]) / (weights_sum ** 2) * \
                            temp_1[k][i] * \
                            mi_fuzzy[k][i] * (1 - mi_fuzzy[k][i]) * (x[k] - self.fuzzy_a[k][i])

                    # 反向传播修正后件参数 需要的偏导数
                    for k in range(self.fuzzy_set + 1):
                        if k == self.fuzzy_set:
                            d_subsequence[k][i] += -(y - o) * weights[i] / weights_sum
                        else:
                            d_subsequence[k][i] += -(y - o) * weights[i] / weights_sum * x[k]

                if (k + 1) % batch_size == 0 or k == len(inputs) - 1:
                    for i in range(self.rule_number):
                        # bp的反向传播更新前件和后件参数
                        for j in range(self.fuzzy_set):  # i代表着第A、B个模糊集
                            self.fuzzy_a[j][i] += -eta[0] * dfuzzy_a[j][i]
                            self.fuzzy_b[j][i] += -eta[0] * dfuzzy_b[j][i]

                        for k in range(self.fuzzy_set + 1):
                            self.subsequence[k][i] += -eta[1] * d_subsequence[k][i]

                        for j in range(self.fuzzy_set):  # i代表着第A、B个模糊集
                            dfuzzy_a[j][i] = 0
                            dfuzzy_b[j][i] = 0

                        for k in range(self.fuzzy_set + 1):
                            d_subsequence[k][i] = 0

            MSE = self.MSE(inputs, labels)
            RMSE = self.RMSE(inputs, labels)
            MAE = self.MAE(inputs, labels)
            R_squared = self.R_squared(inputs, labels)
            Correlation = self.Correlation(inputs, labels)
            Bias = self.Bias(inputs, labels)
            if (epoch + 1) % print_error == 0:  # 每1000次输出均方差
                print("Epoch: %d MSE: %f RMSE: %f MAE: %f R^2: %f Cor: %f Bias: %f" % (epoch + 1, MSE, RMSE, MAE, R_squared, Correlation, Bias))
            self.MSEs.append(MSE)
            self.RMSEs.append(RMSE)
            self.MAEs.append(MAE)
            self.R_squareds.append(R_squared)
            self.Correlations.append(Correlation)
            self.Biass.append(Bias)

    # 测试
    def test(self, inputs, labels):
        MSE = self.MSE(inputs, labels)
        RMSE = self.RMSE(inputs, labels)
        MAE = self.MAE(inputs, labels)
        R_squared = self.R_squared(inputs, labels)
        Correlation = self.Correlation(inputs, labels)
        Bias = self.Bias(inputs, labels)
        print("测试集评估 MSE: %f RMSE: %f MAE: %f R^2: %f Cor: %f Bias: %f" % (
            MSE, RMSE, MAE, R_squared, Correlation, Bias))

    # 均方误差MSE
    def MSE(self, inputs, labels):
        predictions = np.zeros(labels.shape)
        for i in range(inputs.shape[0]):
            predictions[i] = self.output(inputs[i])
        return np.mean((predictions - labels) ** 2)

    # 均方根误差RMSE
    def RMSE(self, inputs, labels):
        predictions = np.zeros(labels.shape)
        for i in range(inputs.shape[0]):
            predictions[i] = self.output(inputs[i])
        return np.sqrt(np.mean((predictions - labels) ** 2))

    # 平均绝对误差MAE
    def MAE(self, inputs, labels):
        predictions = np.zeros(labels.shape)
        for i in range(inputs.shape[0]):
            predictions[i] = self.output(inputs[i])
        return np.mean(np.abs(predictions - labels))

    # R平方
    def R_squared(self, inputs, labels):
        predictions = np.zeros(labels.shape)
        for i in range(inputs.shape[0]):
            predictions[i] = self.output(inputs[i])
        y_bar = np.mean(labels)
        ss_tot = np.sum((labels - y_bar) ** 2)
        ss_res = np.sum((labels - predictions) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        return r_squared

    # 相关系数
    def Correlation(self, inputs, labels):
        predictions = np.zeros(labels.shape)
        for i in range(inputs.shape[0]):
            predictions[i] = self.output(inputs[i])
        corr_mat = np.corrcoef(predictions, labels)
        return corr_mat[0, 1]

    # 偏差
    def Bias(self, inputs, labels):
        predictions = np.zeros(labels.shape)
        for i in range(inputs.shape[0]):
            predictions[i] = self.output(inputs[i])
        residuals = predictions - labels
        mean_residual = np.mean(residuals)
        return mean_residual

    # 预测值与真实值的散点图
    def scatter_plot(self, inputs, labels):
        predictions = np.zeros(labels.shape)
        for i in range(inputs.shape[0]):
            predictions[i] = self.output(inputs[i])
        plt.scatter(predictions, labels)
        plt.xlabel('Predictions')
        plt.ylabel('True Values')
        # 计算拟合的直线
        a, b = np.polyfit(predictions, labels, 1)
        x = np.linspace(min(predictions), max(predictions))
        y = a * x + b
        plt.plot(x, y, '-r')
        plt.show()

    def setup(self, metaheuristics_result):
        self.fuzzy_a = metaheuristics_result[0]
        self.fuzzy_b = metaheuristics_result[1]
        self.subsequence = metaheuristics_result[2]

    # 用以元启发算法优化初始前后件参数
    def metaheuristics_fitness(self, inputs, labels, fuzzy_a, fuzzy_b, subsequence):
        self.fuzzy_a = fuzzy_a
        self.fuzzy_b = fuzzy_b
        self.subsequence = subsequence
        return self.MSE(inputs, labels)


# f = lambda x, y, z: ((x - ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15) ** 2 + (y + 2) ** 2 - 5 * x * y + 3 + z)
# dataset = [((x, y, z), f(x, y, z)) for x in range(-4, 5) for y in range(-4, 5) for z in range(-4, 5)]

f = lambda x, y: ((x - 1) ** 2 + (y + 2) ** 2 - 5 * x * y + 3)
dataset = [((x, y), f(x, y)) for x in range(-4, 5) for y in range(-4, 5)]

x_train = np.array([sub_array[0] for sub_array in dataset])
y_train = np.array([sub_array[1] for sub_array in dataset])

anfis = ANFIS(len(x_train[0]), 5)  # 模糊集的个数和每个模糊集的规则数
anfis.train(x_train, y_train, (0.0001, 0.001), 1, 1000)
anfis.test(x_train, y_train)
anfis.scatter_plot(x_train, y_train)

print(anfis.output((-4, -4), train = False))

plt.figure(figsize = (10, 5))  # 可选，设置画布大小
for client_number in range(noY_client_number):
    client = eval("client_" + str(client_number + 1))
    plt.plot(client.train_loss_history, color = colors[client_number], label = 'client_' + str(client_number + 1) + '\' Train loss', linestyle='--')
    plt.plot(client.test_loss_history, color = colors[client_number], label = 'client_' + str(client_number + 1) + '\' Test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.savefig('Training and Test Loss', bbox_inches = 'tight', pad_inches = 0)
plt.show()


