# -- coding: utf-8 --
# from 算法.meta-anfis.GD_ANFIS_sigmoid隶属度函数 import *
from GD_ANFIS_gaussian隶属度函数 import *
from PSO import *
from GA import *
from SA import *
from ACO import *
import pandas as pd
import os                 # 用GPU跑
import tensorflow as tf   # 用GPU跑
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # 保存模型

# 用GPU跑
def gpu_run():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不关心AVX的支持
    # 设置定量的GPU使用量
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 占用GPU90%的显存
    session = tf.compat.v1.Session(config = config)


if __name__ == '__main__':
    gpu_run()  # GPU训练

    # 创建文件目录
    dirs = 'saveModel'
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    # ===========================数据集===========================

    # # 波士顿房价数据集 404个样本 13个特征 1个标签
    # from sklearn.datasets import load_boston
    # from sklearn.preprocessing import StandardScaler
    # data = load_boston()
    # x = data.data  # 输入数据
    # # 标准化
    # scaler = StandardScaler()
    # x = scaler.fit_transform(x)
    # y = data.target  # 标签
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

    # 非线性函数数据集
    # f = lambda x, y: ((x - ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15) ** 2 + (y + 2) ** 2 - 5 * x * y + 3)
    # dataset = [((x, y), f(x, y)) for x in range(-4, 5) for y in range(-4, 5)]
    # x_train = np.array([sub_array[0] for sub_array in dataset])
    # y_train = np.array([sub_array[ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15] for sub_array in dataset])
    # x_test = np.array([sub_array[0] for sub_array in dataset])
    # y_test = np.array([sub_array[ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15] for sub_array in dataset])

    # 红酒质量 914个样本 11个特征 1个标签
    data = pd.read_csv("dataset/WineQT.csv")
    x = data.drop("quality", axis = 1)
    y = data["quality"].values
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

    # ===========================数据集 尾===========================
    rule = 5  # 自定义调整，找最佳规则数

    # # PSO_ANFIS 3 gaussian 1000 wineQuality
    # pso_anfis_3_gaussian_1000_wineQuality = ANFIS(len(x_train[0]), 3)  # 模糊集的个数和每个模糊集的规则数
    # pso = PSO(200, 100, x_train, y_train, 3, pso_anfis_3_gaussian_1000_wineQuality)
    # joblib.dump(pso, dirs + '/pso_anfisModel/pso_200_3_gaussian_1000_wineQuality_随机初始粒子.dat')
    # pso_anfis_3_gaussian_1000_wineQuality.setup(pso.main())
    # pso_anfis_3_gaussian_1000_wineQuality.train(x_train, y_train, (0.0001, 0.001), ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15, 1000)
    # pso_anfis_3_gaussian_1000_wineQuality.test(x_test, y_test)
    # joblib.dump(pso_anfis_3_gaussian_1000_wineQuality, dirs + '/pso_anfisModel/pso_anfis_200_3_gaussian_1000_wineQuality_随机初始粒子.dat')

    # # SA_ANFIS 3 gaussian 1000 wineQuality
    # sa_anfis_3_gaussian_1000_wineQuality = ANFIS(len(x_train[0]), 3)  # 模糊集的个数和每个模糊集的规则数
    # sa = PSO(100, 100, x_train, y_train, 3, sa_anfis_3_gaussian_1000_wineQuality)
    # joblib.dump(sa, dirs + '/sa_anfisModel/sa_100_3_gaussian_1000_wineQuality_随机初始粒子.dat')
    # sa_anfis_3_gaussian_1000_wineQuality.setup(sa.main())
    # sa_anfis_3_gaussian_1000_wineQuality.train(x_train, y_train, (0.0001, 0.001), 1, 1000)
    # sa_anfis_3_gaussian_1000_wineQuality.test(x_test, y_test)
    # joblib.dump(sa_anfis_3_gaussian_1000_wineQuality, dirs + '/sa_anfisModel/sa_anfis_100_3_gaussian_1000_wineQuality_随机初始粒子.dat')





    # ANFIS
    anfis = ANFIS(len(x_train[0]), rule)  # 模糊集的个数和每个模糊集的规则数
    anfis.train(x_train, y_train, (0.0001, 0.001), 1, 10000)
    anfis.test(x_test, y_test)
    #
    # # SA-ANFIS
    # sa_anfis = ANFIS(len(x_train[0]), rule)
    # sa = SA(200, 100, x_train, y_train, rule, sa_anfis)
    # sa_anfis.setup(sa.main())
    # sa_anfis.train(x_train, y_train, (0.0001, 0.001), ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15, 1000)
    # sa_anfis.test(x_test, y_test)
    #
    # # PSO-ANFIS
    # pso_anfis = ANFIS(len(x_train[0]), rule)
    # pso = PSO(200, 100, x_train, y_train, rule, pso_anfis)
    # pso_anfis.setup(pso.main())
    # pso_anfis.train(x_train, y_train, (0.0001, 0.001), ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15, 1000)
    # pso_anfis.test(x_test, y_test)
    #
    # # GA-ANFIS
    # ga_anfis = ANFIS(len(x_train[0]), rule)
    # ga = GA(0.8, 0.2, 200, 100, x_train, y_train, rule, ga_anfis)
    # ga_anfis.setup(ga.main())
    # ga_anfis.train(x_train, y_train, (0.0001, 0.001), ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15, 1000)
    # ga_anfis.test(x_test, y_test)
    #
    # # ACO-ANFIS
    # aco_anfis = ANFIS(len(x_train[0]), rule)
    # aco = ACO(200, 100, x_train, y_train, rule, aco_anfis)
    # aco_anfis.setup(aco.main())
    # aco_anfis.train(x_train, y_train, (0.0001, 0.001), ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15, 1000)
    # aco_anfis.test(x_test, y_test)
    #
    # print(anfis.output((-4, -4), train = False))
    # print("anfis训练完成 MSE: %f RMSE: %f MAE: %f R^2: %f Cor: %f Bias: %f" % (
    #     anfis.MSEs[-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15], anfis.RMSEs[-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15], anfis.MAEs[-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15], anfis.R_squareds[-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15], anfis.Correlations[-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15], anfis.Biass[-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15]))
    # print(sa_anfis.output((-4, -4), train = False))
    # print("sa-anfis训练完成 MSE: %f RMSE: %f MAE: %f R^2: %f Cor: %f Bias: %f" % (
    #     sa_anfis.MSEs[-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15], sa_anfis.RMSEs[-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15], sa_anfis.MAEs[-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15], sa_anfis.R_squareds[-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15], sa_anfis.Correlations[-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15], sa_anfis.Biass[-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15]))
    # print(pso_anfis.output((-4, -4), train = False))
    # print("pso-anfis训练完成 MSE: %f RMSE: %f MAE: %f R^2: %f Cor: %f Bias: %f" % (
    #     pso_anfis.MSEs[-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15], pso_anfis.RMSEs[-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15], pso_anfis.MAEs[-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15], pso_anfis.R_squareds[-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15], pso_anfis.Correlations[-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15], pso_anfis.Biass[-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15]))
    # print(ga_anfis.output((-4, -4), train = False))
    # print("ga-anfis训练完成 MSE: %f RMSE: %f MAE: %f R^2: %f Cor: %f Bias: %f" % (
    #     ga_anfis.MSEs[-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15], ga_anfis.RMSEs[-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15], ga_anfis.MAEs[-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15], ga_anfis.R_squareds[-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15], ga_anfis.Correlations[-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15], ga_anfis.Biass[-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15]))
    # print(aco_anfis.output((-4, -4), train = False))
    # print("aco-anfis训练完成 MSE: %f RMSE: %f MAE: %f R^2: %f Cor: %f Bias: %f" % (
    #     aco_anfis.MSEs[-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15], aco_anfis.RMSEs[-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15], aco_anfis.MAEs[-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15], aco_anfis.R_squareds[-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15], aco_anfis.Correlations[-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15], aco_anfis.Biass[-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15]))


    # # ANFIS 4 sigmoid 10000 boston
    # anfis_4_sigmoid_10000_bosten = ANFIS(len(x_train[0]), 4)  # 模糊集的个数和每个模糊集的规则数
    # anfis_4_sigmoid_10000_bosten.train(x_train, y_train, (0.0001, 0.001), ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15, 10000)
    # anfis_4_sigmoid_10000_bosten.test(x_test, y_test)
    # joblib.dump(anfis_4_sigmoid_10000_bosten, dirs + '/anfis_4_sigmoid_10000_bosten.dat')
    #
    # # ANFIS 5 sigmoid 10000 boston
    # anfis_5_sigmoid_10000_bosten = ANFIS(len(x_train[0]), 5)  # 模糊集的个数和每个模糊集的规则数
    # anfis_5_sigmoid_10000_bosten.train(x_train, y_train, (0.0001, 0.001), ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15, 10000)
    # anfis_5_sigmoid_10000_bosten.test(x_test, y_test)
    # joblib.dump(anfis_5_sigmoid_10000_bosten, dirs + '/anfis_5_sigmoid_10000_boston.dat')
    #
    # # ANFIS 6 sigmoid 10000 boston
    # anfis_6_sigmoid_10000_bosten = ANFIS(len(x_train[0]), 6)  # 模糊集的个数和每个模糊集的规则数
    # anfis_6_sigmoid_10000_bosten.train(x_train, y_train, (0.0001, 0.001), ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15, 10000)
    # anfis_6_sigmoid_10000_bosten.test(x_test, y_test)
    # joblib.dump(anfis_6_sigmoid_10000_bosten, dirs + '/anfis_6_sigmoid_10000_bosten.dat')
    #
    # # ANFIS 4 gaussian 1000 boston
    # anfis_4_gaussian_1000_bosten = ANFIS(len(x_train[0]), 4)  # 模糊集的个数和每个模糊集的规则数
    # anfis_4_gaussian_1000_bosten.train(x_train, y_train, (0.0001, 0.001), ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15, 1000)
    # anfis_4_gaussian_1000_bosten.test(x_test, y_test)
    # joblib.dump(anfis_4_gaussian_1000_bosten, dirs + '/anfis_4_gaussian_249_bosten.dat')
    #
    # # ANFIS 5 gaussian 1000 boston
    # anfis_5_gaussian_1000_bosten = ANFIS(len(x_train[0]), 5)  # 模糊集的个数和每个模糊集的规则数
    # anfis_5_gaussian_1000_bosten.train(x_train, y_train, (0.0001, 0.001), ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15, 1000)
    # anfis_5_gaussian_1000_bosten.test(x_test, y_test)
    # joblib.dump(anfis_5_gaussian_1000_bosten, dirs + '/anfis_5_gaussian_241_boston.dat')
    #
    # # ANFIS 6 gaussian 1000 boston
    # anfis_6_gaussian_1000_bosten = ANFIS(len(x_train[0]), 6)  # 模糊集的个数和每个模糊集的规则数
    # anfis_6_gaussian_1000_bosten.train(x_train, y_train, (0.0001, 0.001), ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15, 1000)
    # anfis_6_gaussian_1000_bosten.test(x_test, y_test)
    # joblib.dump(anfis_6_gaussian_1000_bosten, dirs + '/anfis_6_gaussian_281_boston.dat')
    #
    # # ANFIS 4 sigmoid 3000 wineQuality
    # anfis_4_sigmoid_3000_wineQuality = ANFIS(len(x_train[0]), 4)  # 模糊集的个数和每个模糊集的规则数
    # anfis_4_sigmoid_3000_wineQuality.train(x_train, y_train, (0.0001, 0.001), ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15, 3000)
    # anfis_4_sigmoid_3000_wineQuality.test(x_test, y_test)
    # joblib.dump(anfis_4_sigmoid_3000_wineQuality, dirs + '/anfis_4_sigmoid_3000_wineQuality.dat')
    #
    # # ANFIS 5 sigmoid 3000 wineQuality
    # anfis_5_sigmoid_3000_wineQuality = ANFIS(len(x_train[0]), 5)  # 模糊集的个数和每个模糊集的规则数
    # anfis_5_sigmoid_3000_wineQuality.train(x_train, y_train, (0.0001, 0.001), ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15, 3000)
    # anfis_5_sigmoid_3000_wineQuality.test(x_test, y_test)
    # joblib.dump(anfis_5_sigmoid_3000_wineQuality, dirs + '/anfis_5_sigmoid_3000_wineQuality.dat')
    #
    # # ANFIS 6 sigmoid 3000 wineQuality
    # anfis_6_sigmoid_3000_wineQuality = ANFIS(len(x_train[0]), 6)  # 模糊集的个数和每个模糊集的规则数
    # anfis_6_sigmoid_3000_wineQuality.train(x_train, y_train, (0.0001, 0.001), ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15, 3000)
    # anfis_6_sigmoid_3000_wineQuality.test(x_test, y_test)
    # joblib.dump(anfis_6_sigmoid_3000_wineQuality, dirs + '/anfis_6_sigmoid_3000_wineQuality.dat')

    # # ANFIS 2 gaussian 1000 wineQuality
    # anfis_2_gaussian_1000_wineQuality = ANFIS(len(x_train[0]), 2)  # 模糊集的个数和每个模糊集的规则数
    # anfis_2_gaussian_1000_wineQuality.train(x_train, y_train, (0.0001, 0.001), ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15, 1000)
    # anfis_2_gaussian_1000_wineQuality.test(x_test, y_test)
    # joblib.dump(anfis_2_gaussian_1000_wineQuality, dirs + '/anfis_2_gaussian_1000_wineQuality.dat')
    #
    # # ANFIS 3 gaussian 1000 wineQuality
    # anfis_3_gaussian_1000_wineQuality = ANFIS(len(x_train[0]), 3)  # 模糊集的个数和每个模糊集的规则数
    # anfis_3_gaussian_1000_wineQuality.train(x_train, y_train, (0.0001, 0.001), ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15, 1000)
    # anfis_3_gaussian_1000_wineQuality.test(x_test, y_test)
    # joblib.dump(anfis_3_gaussian_1000_wineQuality, dirs + '/anfis_3_gaussian_1000_wineQuality.dat')
    #
    # # ANFIS 4 gaussian 1000 wineQuality
    # anfis_4_gaussian_1000_wineQuality = ANFIS(len(x_train[0]), 4)  # 模糊集的个数和每个模糊集的规则数
    # anfis_4_gaussian_1000_wineQuality.train(x_train, y_train, (0.0001, 0.001), ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15, 1000)
    # anfis_4_gaussian_1000_wineQuality.test(x_test, y_test)
    # joblib.dump(anfis_4_gaussian_1000_wineQuality, dirs + '/anfis_4_gaussian_1000_wineQuality.dat')
    #
    # # ANFIS 5 gaussian 1000 wineQuality
    # anfis_5_gaussian_1000_wineQuality = ANFIS(len(x_train[0]), 5)  # 模糊集的个数和每个模糊集的规则数
    # anfis_5_gaussian_1000_wineQuality.train(x_train, y_train, (0.0001, 0.001), ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15, 1000)
    # anfis_5_gaussian_1000_wineQuality.test(x_test, y_test)
    # joblib.dump(anfis_5_gaussian_1000_wineQuality, dirs + '/anfis_5_gaussian_1000_wineQuality.dat')
    #
    # # ANFIS 7 sigmoid 3000 wineQuality
    # anfis_7_sigmoid_3000_wineQuality = ANFIS(len(x_train[0]), 7)  # 模糊集的个数和每个模糊集的规则数
    # anfis_7_sigmoid_3000_wineQuality.train(x_train, y_train, (0.0001, 0.001), ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15, 3000)
    # anfis_7_sigmoid_3000_wineQuality.test(x_test, y_test)
    # joblib.dump(anfis_7_sigmoid_3000_wineQuality, dirs + '/anfis_7_sigmoid_3000_wineQuality.dat')
