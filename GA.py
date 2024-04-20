# -- coding: utf-8 --
from operator import itemgetter
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


# 包装个体和里面的基因个数
class Gene:
    def __init__(self, **data):     # data是形参，封装传进来的内容
        self.__dict__.update(data)      # update根据红体字的变量名包装data，整合字典，{"红体字":data}
        self.size = len(data)   # 基因个数


class GA:
    def __init__(self, CXPB, MUTPB, NGEN, pop_size, inputs, labels, rule, model):
        # 根据需求变化
        self.fuzzy_set = len(inputs[0])  # 模糊集个数
        self.rule = rule                 # 规则个数
        self.model = model               # 当前的模型
        self.inputs = inputs             # bp的特征矩阵
        self.labels = labels             # bp的标签矩阵

        # 参数=[交叉率，变异率，繁殖代数，种群规模，自变量最小值，自变量最大值]
        self.CXPB = CXPB
        self.MUTPB = MUTPB
        self.NGEN = NGEN  # 迭代的代数
        self.pop_size = pop_size  # 种群大小
        # 参数的阈值根据需求变化
        self.popmax = 0.5
        self.popmin = -0.5
        population_num = 2 * self.fuzzy_set * self.rule + (self.fuzzy_set + 1) * self.rule  # 变量个数即可行解向量的参数个数

        # self.bound = [[-3 if i<self.fuzzy_set * self.rule else -3 if i<2 * self.fuzzy_set * self.rule else -50 for i in
        #                range(0, population_num)],
        #               [3 if i<self.fuzzy_set * self.rule else -3 if i<2 * self.fuzzy_set * self.rule else 50 for i in
        #                range(0, population_num)]]

        self.bound = [[-3 if i<self.fuzzy_set * self.rule else 3.9 if i<2 * self.fuzzy_set * self.rule else -1 for i in
                       range(0, population_num)],
                      [3 if i<self.fuzzy_set * self.rule else 4.5 if i<2 * self.fuzzy_set * self.rule else 6 for i in
                       range(0, population_num)]]
        pop_x = np.random.uniform(self.bound[0], self.bound[1], size = (self.pop_size, population_num))
        pop = []
        for i in range(self.pop_size):
            fitness = self.fitness(pop_x[i])  # 计算每个染色体的适应度
            pop.append({'Gene': Gene(data = pop_x[i]), 'fitness': fitness})
        # pop = [{'Gene': <__main__.Gene object at 0x000001F90A319FC8>, 'fitness': ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15}, {'Gene': <__main__.Gene object at 0x000001F90A320808>, 'fitness': ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15}
        self.pop = pop  # 每一条染色体是一个字典，该字典有两个内容，分别是包含基因的Gene类和适应度函数值fitness,Gene里还有个体和个体里基因个数
        self.g_best = self.selectBest(self.pop)  # 当前迭代的最优个体
        self.best = self.selectBest(self.pop)  # 全局迭代的最优个体
        self.popobj = []  # 记录群体最优位置的变化 每次的当次群体最优位置
        self.popobj_z = []  # 记录群体最优位置的变化 每次的总次群体最优位置

    # 适应度函数
    def fitness(self, population):
        fuzzy_a, fuzzy_b, subsequence = toarray(population, self.fuzzy_set, self.rule)
        fitness = self.model.metaheuristics_fitness(self.inputs, self.labels, fuzzy_a, fuzzy_b, subsequence)
        return fitness

    # 挑选出当前代种群中的最好个体作为历史记录
    def selectBest(self, pop):
        # itemgetter()获取对象指定域中的值，e.g: a = [ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15,2,3,4,5],c = itemgetter(0,ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15,2),c(a)显示(ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15, 2, 3)
        # 按key排序，此处是升序
        s_inds = sorted(pop, key = itemgetter("fitness"), reverse = False)
        return s_inds[0]

    # 选择 按照概率从上一代种群中选择个体，直至形成新的一代。我们需要适应度函数值大的个体被选择的概率大，可以使用轮盘赌选择法。该方法的步骤如收藏
    def selection(self, individuals, k):
        """
            从人口中选择一些好的个体，注意好的个体被选择的概率更大
            例如：一个像这样的适应度列表：[5, 4, 3, 2, ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15]，总和是15。
            [-----|----|---|--|-]
            012345|6789|101112|1314|15
            我们在[0, 15]中随机选择一个值。
            第一刻度被选的概率最大
        """
        s_inds = sorted(individuals, key = itemgetter("fitness"), reverse = False)  # 个体按适应度升序排列，后续通过分母化使适合轮盘堵算法
        sum_fits = sum(1/ind['fitness'] for ind in individuals)  # 适应度求总和
        chosen = []
        for i in range(k):
            u = np.random.random() * sum_fits  # 在[0, sum_fits(适应度总和)]的范围内随机产生一个数字，作为阈值。
            sum_ = 0
            for ind in s_inds:
                sum_ += 1/ind['fitness']
                if sum_>=u:     # 前几个个体若适应度相加满足阈值，选择这前几个个体
                    # 搞了半天才明白，是轮盘赌选择，一开始会误以为“只要选择前几个大的不就行了吗”，可能可行，但容易陷入局部最优，不方便交叉的可能性扩展
                    # 因为适应度是降序排列的，虽然每次计算sum_是升序的，但是越靠前的数的sum_占的区间是越大的，所以哪怕u是总适应度范围内随机选的
                    # 靠前的适应度被选的概率还是越大的，举个例子，[5,4,3,2,ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15]去选择，sum_[     5    9   12  14 15]，每个空格对应[5,4,3,2,ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15]
                    # 明显靠前的空格越大，u停留的范围越大，被选的概率也越大
                    chosen.append(ind)
                    break
        chosen = sorted(chosen, key = itemgetter("fitness"), reverse = True)
        return chosen

    # 交叉 两个个体的基因片段在某一点或者某几点进行互换，常用的有单点交叉和双点交叉，两者都很大的改变了原来的基因序列，它是实现优化的重要手段。。它的过程如收藏
    # 实现了双点交叉，其中为了防止只有一个基因的存在，我们使用一个判断语句。
    def crossoperate(self, offspring):  # offspring为待交叉的两个个体
        """
            cross operation
            here we use two points crossoperate
            for example: gene1: [5, 2, 4, 7], gene2: [3, 6, 9, 2], if pos1=ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15, pos2=2
            5 | 2 | 4  7
            3 | 6 | 9  2
            =
            3 | 2 | 9  2
            5 | 6 | 4  7
        """
        dim = len(offspring[0]['Gene'].data)    # 例子这里是4

        geninfo1 = offspring[0]['Gene'].data  # 第一个后代的基因数据
        geninfo2 = offspring[1]['Gene'].data  # 第二个后代的基因数据

        if dim == 1:    # 当基因数只有一位，只能在这一位交叉
            pos1 = 1
            pos2 = 1
        else:
            pos1 = np.random.randint(1, dim)  # 选择一个交叉的位置
            pos2 = np.random.randint(1, dim)

        newoff1 = Gene(data = [])  # 交叉操作产生的后代1
        newoff2 = Gene(data = [])  # 交叉操作产生的后代2
        temp1 = np.array([])
        temp2 = np.array([])
        for i in range(dim):    # pos1左边位交叉，pos2右边位含pos2交叉
            if min(pos1, pos2) <= i < max(pos1, pos2):
                temp2 = np.append(temp2, geninfo2[i])
                temp1 = np.append(temp1, geninfo1[i])
            else:
                temp2 = np.append(temp2, geninfo1[i])
                temp1 = np.append(temp1, geninfo2[i])
        newoff1.data = temp1
        newoff2.data = temp2
        return newoff1, newoff2

    # 变异 变异在遗传过程中属于小概率事件，但是在种群数量较小的情况下，只通过交叉操作并不能产生优秀的后代，此时变异就显得非常重要了。通过适当的变异甚至能够产生更优秀的后代。变异的方式有很多种，常规的变异有基本位变异和逆转变异。它的过程如收藏
    def mutation(self, crossoff):

        dim = len(crossoff.data)

        if dim == 1:
            pos = 0
        else:
            pos = np.random.randint(0, dim)  # 在交叉点选择一个位置进行变异

        crossoff.data[pos] = np.random.rand() - self.popmax

        return crossoff

    def main(self):
        for g in range(self.NGEN):
            # print("############### Generation {} ###############".format(g))
            # 根据转换的适应度做选择
            selectpop = self.selection(self.pop, self.pop_size)
            nextoff = []
            while len(nextoff) != self.pop_size:
                # 对子代进行交叉和变异
                # 选择两个个体
                offspring = [selectpop.pop() for _ in range(2)]  # .pop()删除最后一个元素，for _ in range(n) 一般仅仅用于循环n次，不用设置变量，用 _ 指代临时变量
                if np.random.rand()<self.CXPB:  # 根据概率CXPB交叉两个体
                    crossoff1, crossoff2 = self.crossoperate(offspring)
                    if np.random.rand()<self.MUTPB:  # 根据概率MUTPB变异一个个体
                        muteoff1 = self.mutation(crossoff1)
                        muteoff2 = self.mutation(crossoff2)
                        fit_muteoff1 = self.fitness(muteoff1.data)
                        fit_muteoff2 = self.fitness(muteoff2.data)
                        nextoff.append({'Gene': muteoff1, 'fitness': fit_muteoff1})
                        nextoff.append({'Gene': muteoff2, 'fitness': fit_muteoff2})
                    else:
                        fit_crossoff1 = self.fitness(crossoff1.data)
                        fit_crossoff2 = self.fitness(crossoff2.data)
                        nextoff.append({'Gene': crossoff1, 'fitness': fit_crossoff1})
                        nextoff.append({'Gene': crossoff2, 'fitness': fit_crossoff2})
                else:
                    nextoff.extend(offspring)

            # 个体完全被后代取代
            self.pop = nextoff

            # 列表存放所有适应度
            fits = [ind['fitness'] for ind in self.pop]

            self.g_best = self.selectBest(self.pop)

            if self.g_best['fitness']<self.best['fitness']:
                self.best = self.g_best

            self.popobj.append(self.g_best['fitness'])  # 记录群体最优位置的变化 每次的当次群体最优位置
            self.popobj_z.append(self.best['fitness'])  # 记录群体最优位置的变化 每次的总次群体最优位置
            if (g + 1) % 50 == 0:  # 每50次输出均方差
                print("GA Epoch: %d MSE: %f" % (g + 1, float(self.g_best['fitness'])))

        return toarray(self.best['Gene'].data, self.fuzzy_set, self.rule)
        #     print("Best individual found is {}, {}".format(self.best['Gene'].data, self.best['fitness']))
        #     print("  Max fitness of current pop: {}".format(max(fits)))
        # print("------ End of (successful) evolution ------")
