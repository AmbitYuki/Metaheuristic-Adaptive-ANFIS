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


class PSO:
    def __init__(self, NGEN, pop_size, inputs, labels, rule, model):  # parameters包括 迭代代数 蚁群的个体数量 四个变量的下界 四个变量的下界
        # 根据需求变化
        self.fuzzy_set = len(inputs[0])  # 模糊集个数
        self.rule = rule                 # 规则个数
        self.model = model               # 优化的模型
        self.inputs = inputs             # 输入
        self.labels = labels             # 标签

        self.w = 0.5                # 惯性权重最大值
        self.c1 = 2                 # 个体学习因子
        self.c2 = 2                 # 社会学习因子
        self.maxgen = NGEN          # 进化次数
        self.pop_size = pop_size    # 种群规模

        population_num = 2 * self.fuzzy_set * self.rule + (self.fuzzy_set + 1) * self.rule  # 变量个数即可行解向量的参数个数
        # 记录群体最优位置的变化，即每次全局最优适应度值，用于图像对比
        self.popobj = []  # 记录群体最优位置的变化 每次的当次群体最优位置
        self.popobj_z = []  # 记录群体最优位置的变化 每次的总次群体最优位置
        # self.bound = [[-3 if i<self.fuzzy_set * self.rule else -3 if i<2 * self.fuzzy_set * self.rule else -50 for i in
        #                range(0, population_num)],
        #               [3 if i<self.fuzzy_set * self.rule else 3 if i<2 * self.fuzzy_set * self.rule else 50 for i in
        #                range(0, population_num)]]

        self.bound = [[-1 if i<self.fuzzy_set * self.rule else 3.9 if i<2 * self.fuzzy_set * self.rule else -1 for i in
                       range(0, population_num)],
                      [1 if i<self.fuzzy_set * self.rule else 4.5 if i<2 * self.fuzzy_set * self.rule else 6 for i in
                       range(0, population_num)]]

        # 产生初始粒子位置和速度(随机)
        # self.pop = np.tile(np.array([2.472221787023231, 0.5020198672578563, -0.14677628613192242, -0.4187855575227627, 0.8551072442919343, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.0213400856023525, 0.35176885428487265, -0.07498533290887663, 2.0543198293071114, 0.9727090350111501, ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.1390567652825576, -0.6232667815755891, -0.6199199073533634, -0.5353335881674605, -0.7397276178808149, -0.2799811105867445, -0.572000151996414, -0.5751340629431894, 0.8969092489252735, -0.622339335522355, -9.405648069401346, -5.0212067545956245, -5.215494974381095, 14.350331469344395, -5.4907284789391175, 31.02490122282394, -5.591514941162422, -4.812211690979756, -14.39602137177296, -5.5300186445435795, 11.432850429136337, 3.1897174891711626, 4.110598905751951, 11.575716045069742, 2.133073614381349]), (self.pop_size,ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15))

        # original_array = [
        #     [[-0.6009935853624696, -0.8858232631045155, -0.3557647960803607, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.1382298726048494, 0.6870354245460074],
        #      [-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.9157261926704081, -0.7264919803150316, 0.5644876260898238, -0.2154044584576199, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.0105120306459507],
        #      [-0.5968196311763113, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.0274807663224088, -0.464761517790456, -0.172760302662387, 0.5963426330477114],
        #      [-0.20887657631881548, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.8146279720041087, 0.33169965900138465, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.6066941003751782, -0.3088040429106797],
        #      [-0.5720195057825682, -0.8600386523665734, -0.7808746621834183, 0.786451083934959, 0.4123677907437819],
        #      [0.26949195321796304, 0.7639410628709511, 0.4227857888999035, -0.45745220480405296, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.1739496854975056],
        #      [-0.7477369788071714, -0.562617495268871, -0.8600648893467038, -0.15806563403831728, -0.257453759592708],
        #      [-0.01371423947331191, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.9316101398279049, -0.5594402995248999, -0.5636618946145472, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.1077072500815073],
        #      [-0.6690161901380778, -0.6661688315376134, 0.09254887676800636, -0.6938494550763634, 0.22998870946199165],
        #      [-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.1319611994471706, 0.34948802663423856, -0.44313510413430074, -0.24013533843489157,
        #       0.47427829210502936], [0.03859188350610704, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.0279951006599646, -0.41066152953622365, ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.2471312141145645,
        #                              -0.033448531448845534],
        #      [-0.8032555156263974, -0.7348202012521164, -2.2699762938177455, -0.6740281948696023, -0.38090368676400643],
        #      [-0.5485582425105611, -0.9783449447862722, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.0132955808077648, 0.700314569770564, 0.3467319452361086]],
        #     [[-0.14531650689151718, -0.6251668666631758, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.5211916026312575, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.217817951409827, -0.7166914286352106],
        #      [-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.62466769615774, -0.32090685012867215, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.278972084576685, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.2659423104022842, -0.9121284189992076],
        #      [0.8556287736466897, 0.05504982923393288, -0.6786895462463792, -0.5492803029245701, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.8420861359881238],
        #      [-0.7471184602621285, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.7152774713023984, -0.9662892564294294, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.4032848773716615, -0.851854166069209],
        #      [-0.558328153240635, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.5120363766092755, -3.7977337089443295, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.6626461626174678, -2.3112813561979952],
        #      [-2.6223423331434503, -2.0215570796907616, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.2874171185930512, -0.36887919127678437, -2.7945010203456837],
        #      [-0.8959854089600577, ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.018640706197466, -2.85477063352265, -2.005165694351853, -0.8484109209122694],
        #      [-0.8881045674054746, -2.2498647614205276, 0.6764620523763337, -3.8717970995431354, -0.687514772237455],
        #      [-0.2842211675255615, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.815377466022076, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.4155997552250827, -0.4575036484050177, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.3279931026664744],
        #      [-6.323577555168871, ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.1613098315522632, 0.7543290286047302, -0.8378524164643791, 0.6162736363824345],
        #      [ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.2275358023783343, 0.41745611470189503, -4.367613625619301, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.297655525845483, -2.147876778058266],
        #      [-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.9804759910410716, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.8998625554503965, -5.480540572299144, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.7526273308719407, -0.4118275792776989],
        #      [-2.4670816357790732, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.6552558348093394, 0.31661210126727163, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.322241893867588, -3.482214780501855]],
        #     [[-3.1487072150561217, -5.0874313159765165, 7.82248756823801, -5.3489217545323235, -0.10611582453034099],
        #      [0.020424974784686904, 0.1893193383617484, -5.176125201770133, -0.3511746847836434, -5.68540014703578],
        #      [-2.701438694078861, -0.5815635926177167, ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.917985251326289, -0.12390287605302211, 2.9573031458137837],
        #      [0.9254735273740573, 0.7079105016628723, 3.3805518293537125, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.951971002211873, 0.3507271621646027],
        #      [ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.0727916243072537, -2.2952615318898673, -6.297878037544469, 5.703172780096462, -0.7497546991865628],
        #      [-0.6886647641361543, 9.002059191156508, -8.241742798816665, -0.7304946694499104, 0.19455028302058885],
        #      [2.7293677127903186, 4.792559417178968, 12.015887411799993, -9.077179977631936, ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.448247177607065],
        #      [2.012723065568641, -2.2336022173123555, -12.972070039452221, ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.917833498157685, 9.723019386965875],
        #      [5.183730687805571, -5.950017452631565, 6.078327443075338, 3.3251456438067626, -2.05427592885346],
        #      [-4.879552957152658, 4.28339278300717, 6.661452086740523, -0.40071090414228155, 0.9157426272728131],
        #      [-0.9394916063946472, -0.47014419188646067, -5.697708886953478, 5.830493793264301, -0.04165667795498958],
        #      [4.738440061987264, -4.96589143910563, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.9688892512350424, 6.725223149987102, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.1403869324230003],
        #      [-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.7569328021098143, 2.5236622466149323, 5.646027742917204, 0.3662044440225718, -ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.8956818585410622],
        #      [6.948737883030464, 38.596869821735524, 9.820864927768353, 11.087203053170464, 14.057377058598115]]
        # ]
        # from itertools import chain
        # flattened_original_array = list(chain.from_iterable(original_array))
        # flattened_original_array = np.array(flattened_original_array)
        # flattened_original_array = flattened_original_array.reshape(-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15, ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15)
        # repeated_array = np.tile(flattened_original_array, self.pop_size)
        # repeated_array = repeated_array.reshape(self.pop_size, population_num)
        # self.pop = repeated_array

        self.pop = np.random.uniform(self.bound[0], self.bound[1], size=(self.pop_size, population_num))
        self.v = np.random.uniform(self.bound[0], self.bound[1], size=(self.pop_size, population_num)) * 0.2

        self.fit = []  # 初始化适应度矩阵
        for p in self.pop:
            self.fit.append(self.fitness(p))  # 计算每个粒子的适应度 此时是列表类型
        # numpy.argmin(a, axis=None, out=None)给出axis方向最小值的下标 没有axis则水平展开
        self.i = np.argmin(np.array(self.fit))  # 找最好的个体 即在一维列表里找适应度(误差总值)最低值的下标
        self.gbest = self.pop  # 记录个体最优位置 先初始化gbest
        self.zbest = self.pop[self.i]  # 记录群体最优位置 取第i行的数据初始化zbest 因为第i个粒子适应度最小
        self.fitnessgbest = self.fit  # 个体最佳适应度值 列表
        self.fitnesszbest = self.fit[self.i]  # 全局最佳适应度值

    def fitness(self, population):
        fuzzy_a, fuzzy_b, subsequence = toarray(population, self.fuzzy_set, self.rule)
        fitness = self.model.metaheuristics_fitness(self.inputs, self.labels, fuzzy_a, fuzzy_b, subsequence)
        return fitness

    def main(self):
        for t in range(self.maxgen):
            # 速度更新
            self.v = self.w * self.v + self.c1 * np.random.random() * (self.gbest - self.pop) + self.c2 * np.random.random() * (self.zbest - self.pop)

            # 位置更新
            self.pop = self.pop + 0.5 * self.v

            # 根据更新完的位置重新计算每个粒子的适应度值
            for i in range(len(self.pop)):
                fit = self.fitness(self.pop[i])

                # 个体最优位置更新
                if fit < self.fitnessgbest[i]:
                    self.fitnessgbest[i] = fit

            # 群体最优更新
            j = int(np.argmin(self.fitnessgbest))
            if self.fitnessgbest[j] < self.fitnesszbest:
                self.zbest = self.pop[j]
                self.fitnesszbest = self.fitnessgbest[j]

            self.popobj.append(float(self.fitnessgbest[j]))  # 记录群体最优位置的变化 每次的当次群体最优位置
            self.popobj_z.append(float(self.fitnesszbest))  # 记录群体最优位置的变化 每次的总次群体最优位置
            if (t + 1) % 10 == 0:  # 每10次输出均方差
                print("PSO Epoch: %d MSE: %f" % (t + 1, float(self.popobj[-1])))
            # print(self.pop[0])
        return toarray(self.zbest, self.fuzzy_set, self.rule)
