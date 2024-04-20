# -- coding: utf-8 --
import numpy as np


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


class ACO:
    def __init__(self, NGEN, pop_size, inputs, labels, rule, model):  # parameters包括 迭代代数 蚁群的个体数量 四个变量的下界 四个变量的下界
        # 根据需求变化
        self.fuzzy_set = len(inputs[0])  # 模糊集个数
        self.rule = rule                 # 规则个数
        self.model = model               # 优化的模型
        self.inputs = inputs             # 输入
        self.labels = labels             # 标签

        # 初始化
        self.NGEN = NGEN  # 迭代的代数
        self.pop_size = pop_size  # 种群大小
        self.population_num = 2 * self.fuzzy_set * self.rule + (self.fuzzy_set + 1) * self.rule  # 变量个数即可行解向量的参数个数
        self.bound = [[-3 if i < self.fuzzy_set * self.rule else -3 if i < 2*self.fuzzy_set * self.rule else -50 for i in range(0, self.population_num)], [3 if i < self.fuzzy_set * self.rule else 3 if i < 2*self.fuzzy_set * self.rule else 50 for i in range(0, self.population_num)]]

        self.pop = np.random.uniform(self.bound[0], self.bound[1], size=(self.pop_size, self.population_num))
        self.g_best = np.zeros((1, self.population_num))[0]  # 当前迭代的全局蚂蚁最优的位置，大小等于一个个体
        self.best = self.pop[0]  # 所有迭代的全局蚂蚁最优的位置
        self.popobj = []  # 记录群体最优位置的变化 每次的当次群体最优位置
        self.popobj_z = []  # 记录群体最优位置的变化 每次的总次群体最优位置

        # 初始化第0代初始全局最优解
        # 根据需求变的
        temp = 10000    # 与第1次迭代的全局蚂蚁最优位置适应度比较，因为是损失函数为目标函数，所以比较参照数要设大

        for i in range(self.pop_size):  # 遍历每个蚂蚁索引，即遍历蚂蚁位置矩阵的行
            # for j in range(self.var_num):  # 对2维列表pop_x做遍历，每一行可以视作一个个体，每个个体里面含一组完整的解，此处即遍历蚂蚁位置矩阵的列
            #     self.pop_x[i][j] = np.random.uniform(self.bound[0][j], self.bound[ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15][j])  # 为每一个蚂蚁的每一个解随机生成一个答案，答案在bound对应位置的范围里，numpy.random.uniform(low=0.0, high=ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.0, size=None)
            fit = self.fitness(self.pop[i])  # 为每个个体计算适应度
            if fit < temp:  # 寻找当前种群的最优个体，初始化g_best即为pop_x里适应度最优的
                self.g_best = self.pop[i]  # gbest为最优位置列表更新最大值，即最优个体
                temp = fit  # 同时刷新最大值的比较标准

    def fitness(self, population):
        fuzzy_a, fuzzy_b, subsequence = toarray(population, self.fuzzy_set, self.rule)
        fitness = self.model.metaheuristics_fitness(self.inputs, self.labels, fuzzy_a, fuzzy_b, subsequence)
        return fitness

    def update_operator(self, gen, t, t_max):   # 当前迭代次数，信息素列表(适应度列表)，最大信息素(最大适应度)
        """
        更新算子：根据概率更新下一时刻的位置和信息素，挑选最好的位置保存
        gen是当前的代数，t是信息素列表，t_max是当前信息素列表中信息素最大的值
        每个个体都对应一个信息素量，当信息素相对少时该个体便大概率进行行动，迭代次数多了之后个体的优度整体提升
        """
        rou = 0.8  # 信息素挥发系数
        Q = 1  # 信息释放总量，蚂蚁们工作循环一次释放的信息总量
        lamda = 1 / gen  # lamda随着代数增加而减小，用于局部搜索
        pi = np.zeros(self.pop_size)  # 概率表，存储每个蚂蚁个体的转移概率，这里由pop_size个0组成的一维矩阵，概率越高，越多蚂蚁涌入那里
        for i in range(self.pop_size):  # 对每一个变量做遍历
            # print("第%d只蚂蚁" % i)
            pi[i] = (t_max - t[i]) / t_max  # 计算行动概率，信息素越少行动概率越大
            for j in range(self.population_num):
                # print(pi[i])
                # 更新蚂蚁们位置
                if pi[i]<0.05:  # 进行局部搜索
                    self.pop[i][j] = self.pop[i][j] + np.random.uniform(-1, 1) * 0.1 * lamda
                else:  # 进行全局搜索
                    self.pop[i][j] = self.pop[i][j] + np.random.uniform(-1, 1) * (
                            self.bound[1][j] - self.bound[0][j]) / 2
                # 越界保护，令每个解的值不会超过边界
                if self.pop[i][j]<self.bound[0][j]:
                    self.pop[i][j] = self.bound[0][j]
                if self.pop[i][j]>self.bound[1][j]:
                    self.pop[i][j] = self.bound[1][j]
            # 更新t值，根据当前的信息素更新下一时刻的信息素
            t[i] = (1 - rou) * t[i] + Q / self.fitness(self.pop[i])
            # 更新全局最优值
            if self.fitness(self.pop[i]) < self.fitness(self.g_best):
                self.g_best = self.pop[i]
        t_max = np.max(t)  # 对信息素序列进行检索得到最大值
        return t_max, t

    def main(self):  # 运行的主程序
        for gen in range(1, self.NGEN + 1):  # 迭代循环
            if gen == 1:  # 第一代首先初始化信息素列表与信息素最大值，直接使用最初的适应度带入计算
                # np.array(list(map(self.fitness, self.pop_x)))为1维200列的适应度矩阵，即各个蚂蚁的y值
                # np.max(np.array(list(map(self.fitness, self.pop_x))))为蚂蚁中最大的适应度
                tmax, t = self.update_operator(gen, np.array(list(map(self.fitness, self.pop))),  # map(function, iterable, ...)
                                               np.max(np.array(list(map(self.fitness, self.pop)))))
            else:  # 第二代之后循环
                tmax, t = self.update_operator(gen, t, tmax)

            # 每单次循环的最佳值 可把self.g_best替换，参考SA
            # print('############ Generation {} ############'.format(str(gen)))  # 打印每代信息
            if self.fitness(self.g_best)<self.fitness(self.best):
                self.best = self.g_best.copy()
            # if gen == self.NGEN // 2:   # ACO可能初始化的权重排列就是所有迭代较优的，导致早期会出现峰段，故对适应度以小为优的标准，加这句判断
            #     # if self.fitness(self.g_best)<self.fitness(self.best):  # self.g_best是当前迭代完的最优位置
            #
            #         self.best = self.g_best.copy()
            # if gen > self.NGEN // 2:  # ACO可能初始化的权重排列就是所有迭代较优的，导致早期会出现峰段，故对适应度以小为优的标准，加这句判断
            #
            #     if self.fitness(self.g_best)<self.fitness(self.best):  # self.g_best是当前迭代完的最优位置
            #         self.best = self.g_best.copy()

            self.popobj.append(self.fitness(self.g_best))  # 记录群体最优位置的变化 每次的当次群体最优位置
            self.popobj_z.append(self.fitness(self.best))  # 记录群体最优位置的变化 每次的总次群体最优位置
            # print('最好的位置：{}'.format(best))
            # print('最大的函数值：{}'.format(self.fitness(best)))
            if (gen + 1) % 50 == 0:  # 每50次输出均方差
                print("ACO Epoch: %d MSE: %f" % (gen + 1, self.fitness(self.g_best)))
        return toarray(self.best, self.fuzzy_set, self.rule)
