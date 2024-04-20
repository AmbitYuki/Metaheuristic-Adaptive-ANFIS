# -*- coding: utf-8 -*-
# import numpy as np
# # def cross_entropy(y_true,y_pred):
# #     C=0
# #     # one-hot encoding
# #     for col in range(y_true.shape[-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15]):
# #         y_pred[col] = y_pred[col] if y_pred[col] < ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15 else 0.99999
# #         y_pred[col] = y_pred[col] if y_pred[col] > 0 else 0.00001
# #         C+=y_true[col]*np.log(y_pred[col])+(ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15-y_true[col])*np.log(ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15-y_pred[col])
# #     return -C
# def cross_entropy(labels, logits):
#     eps = 1e-10
#     predicted_prob = np.clip(logits, eps, ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15 - eps)
#     loss = -np.mean(np.sum(labels * np.log(predicted_prob) + (ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15-labels) * np.log(ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15 - predicted_prob), axis=-ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15))
#     return loss
#
# # 没有考虑样本个数 默认=ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15
# num_classes = 3
# label=ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15#设定是哪个类别 真实值
#
# y_true = np.zeros((num_classes))
# # y_pred = np.zeros((num_classes))
# # preset
# y_true[label]=ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15
# y_pred = np.array([0.0,ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15.0,0.0])
# C = cross_entropy(y_true,y_pred)
# print(y_true,y_pred,"loss:",C)
# y_pred = np.array([0.ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15,0.8,0.ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15])
# C = cross_entropy(y_true,y_pred)
# print(y_true,y_pred,"loss:",C)
# y_pred = np.array([0.2,0.6,0.2])
# C = cross_entropy(y_true,y_pred)
# print(y_true,y_pred,"loss:",C)
# y_pred = np.array([0.3,0.4,0.3])
# C = cross_entropy(y_true,y_pred)
# print(y_true,y_pred,"loss:",C)
# y_pred = np.array([0.8,0.ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15,0.ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15])
# C = cross_entropy(np.array([ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15, 0, 0]),y_pred)
# print(np.array([ANN_自适应PSO——Diagnosis_iid_2noY_14,34_5_15, 0, 0]),y_pred,"loss:",C)


import numpy as np
from keras.datasets import mnist
from sklearn.utils import shuffle

# load mnist data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# preprocess data
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

# shuffle data
X_train, y_train = shuffle(X_train, y_train, random_state = 0)
X_test, y_test = shuffle(X_test, y_test, random_state = 0)

# define model
np.random.seed(0)
n_input = 784
n_hidden = 100
n_output = 10

W1 = np.random.normal(0, 0.01, (n_input, n_hidden))
W2 = np.random.normal(0, 0.01, (n_hidden, n_output))
b1 = np.zeros((1, n_hidden))
b2 = np.zeros((1, n_output))


# define softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis = -1, keepdims = True)


# define categorical cross entropy loss function
def categorical_cross_entropy(logits, labels):
    return -np.mean(np.sum(labels * np.log(logits), axis = -1))


# define derivative of categorical cross entropy loss function
def categorical_cross_entropy_derivative(logits, labels):
    return - labels / logits


# train model
epochs = 10
learning_rate = 0.01
for epoch in range(epochs):
    for i in range(len(X_train)):
        x, y = X_train[i], y_train[i]

        # forward pass
        a1 = x.dot(W1) + b1
        z1 = np.maximum(a1, 0)
        a2 = z1.dot(W2) + b2
        y_pred = softmax(a2)

        # backward pass
        dL = y_pred - y
        dW2 = z1.T.dot(dL)
        db2 = np.sum(dL, axis=0, keepdims=True)
        dz1 = dL.dot(W2.T)
        dz1[a1 <= 0] = 0
        dW1 = np.dot(x.reshape(1,-1).T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # update weights
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    # evaluate model on train data
    a1 = X_train.dot(W1) + b1
    z1 = np.maximum(a1, 0)
    a2 = z1.dot(W2) + b2
    y_pred = softmax(a2)
    train_loss = categorical_cross_entropy(y_pred, y_train)

    # evaluate model on test data
    a1 = X_test.dot(W1) + b1
    z1 = np.maximum(a1, 0)
    a2 = z1.dot(W2) + b2
    y_pred = softmax(a2)
    test_loss = categorical_cross_entropy(y_pred, y_test)

    # print train and test loss
    print("Epoch: {}, Train Loss: {:.4f}, Test Loss: {:.4f}".format(epoch + 1, train_loss, test_loss))
