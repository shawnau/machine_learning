# -*- coding: utf-8 -*-

import adaboost as ab
import test_toolkit as tt
import plot_roc as plot
import json
import numpy as np


def save_model(m):
    weights_json = json.dumps(m.model_weights.tolist(), indent=4, separators=(',', ': '))
    models_json = json.dumps(m.model_list, indent=4, separators=(',', ': '))
    with open("model/model_weights.json", 'w') as fp:
        fp.write(weights_json)
    with open("model/model_list.json", 'w') as fp:
        fp.write(models_json)


def load_model():
    with open("model/pretrained_model_weights.json") as fp:
        weights_json = json.loads(fp.read())
        weights_list = np.array(weights_json)
        size = weights_list.shape[0]
        model = ab.Model(size)
        model.model_weights = weights_list

    with open("model/pretrained_model_list.json") as fp:
        model_list = json.loads(fp.read())
        model.model_list = model_list
    return model

# 载入数据
data_matrix = tt.load_data('test_data/horseColicTest2.txt')

# 预处理, 测试集/训练集分为8:2
train_matrix, cv_matrix, test_matrix = tt.split_data(data_matrix, (0.8, 0.0))
train_matrix_x, class_vector = tt.separate_x_y(train_matrix)

# ------------------1.自己训练模型----------------------
# m = ab.adaboost_train(train_matrix_x, class_vector)

# -----------------2.载入预训练模型----------------------
m = load_model()

# -------------------测试模型泛化能力---------------------
test_matrix_x, test_labels = tt.separate_x_y(test_matrix)
results = ab.adaboost_test(test_matrix_x, test_labels, m)

# 绘制ROC曲线
plot.plot_roc(results, test_labels)

# 保存模型
save_model(m)
