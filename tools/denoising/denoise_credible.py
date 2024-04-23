import numpy as np
from sklearn import preprocessing
from scipy.spatial.distance import cdist
import math
import argparse
import sys

sys.path.append('./')


def main(args):
    SD = args.SD

    SD_tail = SD.split('.')[0].split('_')[-1]
    save_path = './feature_save/'
    if SD_tail == 'class100':
        save_path = save_path + 'EXclass/'
    elif SD_tail == 'data100':
        save_path = save_path + 'EXdata/'
    elif SD_tail == 'class70':
        save_path = save_path + 'OPclass/'
    elif SD_tail == 'data70':
        save_path = save_path + 'OPdata/'
    elif SD_tail == 'class30':
        save_path = save_path + 'OPclass/'
    elif SD_tail == 'data30':
        save_path = save_path + 'OPdata/'

    data = np.load(save_path + f'{SD}_feature.npy')
    labels = np.load(save_path + f'{SD}_label.npy')

    print('data_shape: ', data.shape)
    print('label_shape: ', labels.shape)
    data = preprocessing.normalize(data)
    data = preprocessing.normalize(data, axis=0)

    guard = 0
    count_pre = 0
    count_last = 0
    credible_label = []

    for label in labels:
        if label != guard and count_last == len(labels) - 1:
            assert len(set(labels[count_pre:count_last])) == 1
            avg_feature = np.sum(data[count_pre:count_last], axis=0) / (count_last - count_pre)
            avg_feature = avg_feature.reshape(1, -1)

            dis = cdist(data[count_pre:count_last], avg_feature).reshape(-1)
            sort_dis = np.argsort(dis)
            true_dis = sort_dis[math.ceil(args.T * len(dis)):].tolist()
            dis = np.array([True] * len(dis))
            if true_dis != []:
                dis[true_dis] = False
            credible_label += dis.tolist()
            credible_label += [True]

        elif count_last == len(labels) - 1:
            assert len(set(labels[count_pre:count_last + 1])) == 1
            avg_feature = np.sum(data[count_pre:count_last + 1], axis=0) / (count_last - count_pre)
            avg_feature = avg_feature.reshape(1, -1)

            dis = cdist(data[count_pre:count_last], avg_feature).reshape(-1)
            sort_dis = np.argsort(dis)
            true_dis = sort_dis[math.ceil(args.T * len(dis)):].tolist()
            dis = np.array([True] * len(dis))
            if true_dis != []:
                dis[true_dis] = False
            credible_label += dis.tolist()

        elif label != guard:
            assert len(set(labels[count_pre:count_last])) == 1
            avg_feature = np.sum(data[count_pre:count_last], axis=0) / (count_last - count_pre)
            avg_feature = avg_feature.reshape(1, -1)

            # Denoising Operation
            dis = cdist(data[count_pre:count_last], avg_feature).reshape(-1)
            sort_dis = np.argsort(dis)
            true_dis = sort_dis[math.ceil(args.T * len(dis)):].tolist()
            dis = np.array([True] * len(dis))
            if true_dis != []:
                dis[true_dis] = False
            credible_label += dis.tolist()

            count_pre = count_last
            guard = label
        count_last += 1

    np.save(save_path + f'{SD}_credible_ratio_90_feature.npy', credible_label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Denoising and get credible samples.')
    parser.add_argument('--T', type=float,
                        help='threthold')
    parser.add_argument('--SD', type=str,
                        help='Scenarios detail, For example: f512_r18_arc_class100, f128_r18_softmax_class100')
    main(parser.parse_args())
