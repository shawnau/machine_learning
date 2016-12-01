# -*- coding: utf-8 -*-

import numpy as np


def stump_classifier(data_matrix, feature_index, threshold, rule='lt'):
    """
    :param: data_matrix: 测试集, 样本按行排列
    :param: feature_index: 用来分类的特征序号
    :param: threshold: 阈值
    :param: rule: 规则, 默认情况下是小于阈值被分类为-1.0
    决策树桩
    输入: 测试集
    输出: 预测结果(以1,-1标注)
    """
    results = np.ones((data_matrix.shape[0], 1))
    feature_values = data_matrix[:, feature_index]
    if rule == 'lt':
        results[np.where(feature_values <= threshold)[0]] = -1.0
    elif rule == 'gt':
        results[np.where(feature_values > threshold)[0]] = -1.0
    else:
        print('ERROR: rule not recognized, use default as lt.')
        results[np.where(feature_values <= threshold)[0]] = -1.0
    return results


# 决策树桩, 用于连续特征值数据的二分类预测,
def create_stump(data_matrix, labels, data_weights, step_number=10):
    """
    :param: data_matrix: 测试集, 样本按行排列
    :param: labels: 标注
    :param: data_weights: 训练集样本权重
    :param: step_number: 迭代次数, 亦即设置阈值每一步的步长
    :return: stump: 决策树桩, 用dict实现
    :return: return_prediction: 预测的标注值
    :return: min_error: 最小损失函数值
    决策树桩训练函数
    输入: 训练集, 训练集权重, 迭代次数
    输出: 决策树桩, 输出值, 最小损失
    """
    m, n = np.shape(data_matrix)
    stump = {}
    return_prediction = np.zeros((m, 1))
    min_error = np.inf

    for i in range(n):
        min_value = data_matrix[:, i].min()
        max_value = data_matrix[:, i].max()
        step_size = (max_value - min_value) / float(step_number)
        for j in range(-1, step_number + 1):
            for rule in ['lt', 'gt']:
                threshold = min_value + j * step_size
                prediction = stump_classifier(data_matrix, i, threshold, rule)
                # is_error用来存放是否错误的标记, 即I(prediction = labels)
                is_error = np.ones((m, 1))
                is_error[prediction == labels] = 0
                # 损失乘以归一化的权重
                weighted_error = np.dot(data_weights.T, is_error)
                if weighted_error < min_error:
                    min_error = weighted_error
                    return_prediction = prediction.copy()
                    stump['feature_index'] = i
                    stump['threshold'] = threshold
                    stump['rule'] = rule
    return stump, return_prediction, min_error
