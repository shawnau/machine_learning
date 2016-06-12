import numpy as np


def plot_roc(predict_vector, class_vector):
    import matplotlib.pyplot as plt
    positive_num = np.where(class_vector == 1.0)[0].shape[0]
    negative_num = class_vector.shape[0] - positive_num
    y_sum = 0.0
    cursor = (1.0, 1.0)
    tp_step = 1 / float(positive_num)
    fp_step = 1 / float(negative_num)
    sorted_indices = np.argsort(predict_vector, axis=0)

    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sorted_indices.tolist():
        if class_vector[index] == 1.0:
            delta_fp = 0
            delta_tp = tp_step
        else:
            delta_fp = fp_step
            delta_tp = 0
            y_sum += cursor[1]
        ax.plot([cursor[0], cursor[0] - delta_fp], [cursor[1], cursor[1] - delta_tp], c='b')
        cursor = (cursor[0] - delta_fp, cursor[1] - delta_tp)
    ax.plot([0, 1], [0, 1], 'b--')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print('AUC is: ', y_sum*fp_step)
