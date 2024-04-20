# -- coding: utf-8 --
import numpy as np
import math
"""
    初始温度和终止温度很讲究，两者决定自变量的更新步长
        e.g.权重范围[-0.ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15,0.4]
            (random() - random()范围大概是[-0.5,0.5]
            更新公式为 原始自变量 + T * (np.random.rand() - np.random.rand())
            主观考虑 最大的权重更新步长为0.25/-0.25适宜
            初始最高温度=0.25/0.5=-0.25/-0.5=0.5
            同理最小权重更新步长=0.05
    自变量更新式一定要(np.random.rand() - np.random.rand())，因为向上向下均可调整，光rand()只会向上调整
    跳出局部最优概率式 可以适当根据实际情况添加倍数，比如差值几乎趋于0.00?，而温度趋于0.0?，可以加10倍，一般差值和温度是同比递减的，倍数可以不变
"""


# 权重list转为权重array
def toarray(population, fuzzy_set, rule):
    fuzzy_a = list(population[0:fuzzy_set*rule])
    temp_list = [fuzzy_a[i:i+rule] for i in range(0, len(fuzzy_a), rule)]
    fuzzy_a[:] = temp_list

    fuzzy_b = list(population[fuzzy_set*rule:2*fuzzy_set*rule])
    temp_list = [fuzzy_b[i:i + rule] for i in range(0, len(fuzzy_b), rule)]
    fuzzy_b[:] = temp_list

    subsequence = list(population[2*fuzzy_set*rule:])
    temp_list = [subsequence[i:i + rule] for i in range(0, len(subsequence), rule)]
    subsequence[:] = temp_list
    return fuzzy_a, fuzzy_b, subsequence


class SA:
    def __init__(self, T, pop_size, inputs, labels, rule, model):    # iter内循环迭代次数，T0初始温度，Tf终止温度
        # 根据需求变化
        self.fuzzy_set = len(inputs[0])  # 模糊集个数
        self.rule = rule                 # 规则个数
        self.model = model               # 优化的模型
        self.inputs = inputs             # 输入
        self.labels = labels             # 标签

        self.pop_size = pop_size  # 内循环迭代次数,即为L =100
        self.alpha = 0.95  # 降温系数，alpha=0.99
        self.T0 = T  # 初始温度T0为100
        self.Tf = 0.01  # 温度终值Tf为0.01
        self.T = self.T0  # 当前温度

        population_num = 2 * self.fuzzy_set * self.rule + (self.fuzzy_set + 1) * self.rule  # 变量个数即可行解向量的参数个数
        # self.bound = [[-3 if i<self.fuzzy_set * self.rule else -3 if i<2 * self.fuzzy_set * self.rule else -50 for i in
        #                range(0, population_num)],
        #               [3 if i<self.fuzzy_set * self.rule else 3 if i<2 * self.fuzzy_set * self.rule else 50 for i in
        #                range(0, population_num)]]

        self.bound = [[-1 if i<self.fuzzy_set * self.rule else 3.9 if i<2 * self.fuzzy_set * self.rule else -1 for i in
                       range(0, population_num)],
                      [1 if i<self.fuzzy_set * self.rule else 4.5 if i<2 * self.fuzzy_set * self.rule else 6 for i in
                       range(0, population_num)]]
        self.pop = np.random.uniform(self.bound[0], self.bound[1], size=(self.pop_size, population_num))
        # 记录群体最优位置的变化，即每次全局最优适应度值，用于图像对比
        self.popobj = []  # 记录群体最优位置的变化 每次的当次群体最优位置
        self.popobj_z = []  # 记录群体最优位置的变化 每次的总次群体最优位置

        self.fit = []  # 初始化适应度矩阵
        for p in self.pop:
            self.fit.append(self.fitness(p))  # 计算每个粒子的适应度 此时是列表类型

        # self.i = np.argmin(np.array(self.fit))  # 找最好的个体 即在一维列表里找适应度(误差总值)最低值的下标
        self.zbest = self.pop[0]  # 记录所有次外循环最优权重排列 取第i行的数据初始化zbest 因为第i个粒子适应度最小
        self.fitnessgbest = self.fit  # 记录每次外循环最佳适应度值(临时保存的，保存单次外循环更新后的pop_size个适应度值)
        self.fitnesszbest = self.fit[0]  # 所有次外循环的最佳适应度值(1个，临时保存的，每次外循环会更新)
        self.T_history = []  # 每次外循环的温度记录，用以可视化x轴

    def generate_new(self, pop):  # 扰动产生新解的过程
        pop_new = np.array([i + self.T / 10 * (np.random.rand() - np.random.rand()) for i in pop])  # 一定要(random-random)，不然数值不会向下更新
        return pop_new

    def Metrospolis(self, f, f_new):  # Metropolis准则，判断是否被覆盖
        if f_new<=f:
            return 1
        else:
            # print(("%.2f" % ((f - f_new) * 10)), self.T)
            p = math.exp(((f - f_new) * 10) / self.T)    # 70根据实际情况加，因为两loss的差值很小，以至于e^x每次都接近1
            # print((f - f_new) * 800, self.T)
            # print("%.2f" % p)
            if np.random.rand()<p:
                return 1
            else:
                return 0

    def fitness(self, population):
        fuzzy_a, fuzzy_b, subsequence = toarray(population, self.fuzzy_set, self.rule)
        fitness = self.model.metaheuristics_fitness(self.inputs, self.labels, fuzzy_a, fuzzy_b, subsequence)
        return fitness

    def main(self):
        # print(self.pop)
        count = 0
        # 外循环迭代，当前温度小于终止温度的阈值
        while self.T>self.Tf:
            # print("第{}次迭代".format(count))
            self.fitnessgbest = [10000 for i in range(self.pop_size)]
            # 内循环迭代100次
            for i in range(self.pop_size):
                f = self.fitness(self.pop[i])  # f为迭代一次后的值
                pop_new = self.generate_new(self.pop[i])  # 产生新解
                pop_new[pop_new>self.bound[1][0]] = self.bound[1][0]  # 限制位置
                pop_new[pop_new<self.bound[0][0]] = self.bound[0][0]
                f_new = self.fitness(pop_new)  # 产生新值
                if self.Metrospolis(f, f_new):  # 判断是否接受新值
                    self.pop[i] = pop_new  # 如果接受新值，则把新值的x,y存入x数组和y数组

                # 迭代当前次外循环在该温度下最优适应度
                fit = self.fitness(self.pop[i])
                if fit < self.fitnessgbest[i]:
                    self.fitnessgbest[i] = fit

            # 更新已完成的所有次外循环的最优权重排列和最优适应度值
            j = int(np.argmin(self.fitnessgbest))
            if self.fitnessgbest[j] < self.fitnesszbest:
                self.zbest = self.pop[j]
                self.fitnesszbest = self.fitnessgbest[j]

            self.T_history.append(self.T)
            # 温度按照一定的比例下降（冷却）
            self.T = self.T * self.alpha
            count += 1

            self.popobj.append(float(self.fitnessgbest[j]))  # 记录群体最优位置的变化 每次的当次群体最优位置
            self.popobj_z.append(float(self.fitnesszbest))  # 记录群体最优位置的变化 每次的总次群体最优位置
            if (count + 1) % 50 == 0:  # 每50次输出均方差
                print("SA Epoch: %d MSE: %f" % (count + 1, float(self.fitnessgbest[j])))
        return toarray(self.zbest, self.fuzzy_set, self.rule)
