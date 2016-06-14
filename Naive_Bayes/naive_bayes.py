import numpy as np


def create_dataset():
    text_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    text_class = [0, 1, 0, 1, 0, 1]
    return text_list, text_class


def create_vocab_dic(text_list):
    vocab_set = set([])
    for line in text_list:
        vocab_set = vocab_set | set(line)
    return list(vocab_set)


def words_to_vector(text_vector, vocab_list):
    n = len(vocab_list)
    feature_vector = [0] * n
    for word in text_vector:
        if word in vocab_list:
            feature_vector[vocab_list.index(word)] = 1
    return feature_vector


def p_class(text_class):
    class_set = list(set(text_class))
    n = len(text_class)
    k = len(class_set)
    p_list = [0] * k
    for i in range(k):
        p_list[i] = text_class.count(class_set[i]) / float(n)
    return p_list, class_set


def cond_p(feature_matrix, text_class, x_value, x_feature_index, class_index):
    class_set = list(set(text_class))
    y_indices = np.nonzero(np.array(text_class) == class_set[class_index])[0]
    x_feature_list = list(feature_matrix[y_indices, x_feature_index])
    return x_feature_list.count(x_value) / float(len(x_feature_list))


def naive_bayes(text_list, text_class, input_text):
    vocab_list = create_vocab_dic(text_list)
    # create the matrix of converted feature vectors
    feature_matrix = []
    for line in text_list:
        feature_matrix.append(words_to_vector(line, vocab_list))
    feature_matrix = np.array(feature_matrix)
    # calculate p(c_k)
    p_k, class_set = p_class(text_class)
    input_vector = words_to_vector(input_text, vocab_list)
    m, n = np.shape(feature_matrix)
    k = len(class_set)
    # find the c_k which made the maximum prob
    max_prob = 0.0
    predict_index = 0
    for class_index in range(k):
        production = 1.0
        for j in range(n):
            production *= cond_p(feature_matrix, text_class, input_vector[j], j, class_index)
        prob = p_k[class_index] * production
        if prob > max_prob:
            predict_index = class_index
            max_prob = prob
    return class_set[predict_index]