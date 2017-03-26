# -*- coding: utf-8 -*-

import math
import numpy as np
import stump_tree as st


class Model:
    """
    :return: model_weights: np数组,弱分类器权重
    :return: model_list: weak model list, 其中每个弱分类器用dict实现
    """
    def __init__(self, size=10):
        self.model_weights = np.zeros((size, 1))
        self.model_list = []


def adaboost_train(data_matrix, labels, iteration=40):
    """
    :param: data_matrix: (m,n) np数组, 训练集, 样本按行排列
    :param: labels: (m,1) np数组 标注
    :param: iteration: int 弱分类器个数
    输入训练集和弱分类器个数, 输出模型
    """
    number = data_matrix.shape[0]
    data_weights = np.ones((number, 1)) / number        # 初始化训练集权重为1/number
    m = Model(iteration)                                # 初始化模型权重为0
    for i in range(iteration):
        stump, predictions, weighted_error = st.create_stump(data_matrix, labels, data_weights)
        m.model_list.append(stump)
        m.model_weights[i] = 0.5 * math.log((1.0 - weighted_error) / max(weighted_error, 1e-16))
        data_weights = data_weights * np.exp(-1.0 * m.model_weights[i] * labels * predictions)
        data_weights /= np.sum(data_weights)
    return m


def adaboost_classify(input_matrix, m):
    """
    :param: data_matrix: (m,n) np数组,测试集, 样本按行排列
    :param: m: 模型
    :return: models_output: (m,1) np数组,强分类器输出值
    ensemble model, 输入训练集, 返回输出结果
    """
    models_output = np.zeros((input_matrix.shape[0], 1))
    for i in range(len(m.model_list)):
        model_prediction = st.stump_classifier(input_matrix,
                                               m.model_list[i]['feature_index'],
                                               m.model_list[i]['threshold'],
                                               m.model_list[i]['rule'])
        models_output += m.model_weights[i] * model_prediction
    return np.sign(models_output)


def adaboost_test(data_matrix, labels, m):
    """
    :param: data_matrix: 测试集, 样本按行排列
    :param: labels: 标注
    输入测试集和模型, 输出模型参数, 输出结果和正确率, 返回输出结果
    """
    models_output = adaboost_classify(data_matrix, m)
    i_vec = (np.sign(models_output) == labels).astype(int)
    error_rate = 1 - np.count_nonzero(i_vec) / float(data_matrix.shape[0])
    print '\n'+'models_output:'+'\n', models_output.T, \
          '\n' + 'labels:' + '\n', labels.T, \
          '\n'+'error rate: ', error_rate
    return models_output
