import numpy as np
from sklearn import preprocessing
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
    o_label = np.load(save_path + f'{SD}_label.npy')
    print('loaded')
    sort_index = np.argsort(o_label)
    print('sorted')
    features = data[sort_index]
    labels = o_label[sort_index]
    print(labels.shape)

    avg_features = []
    guard = 0
    count_pre = 0
    count_last = 0
    count = 0
    for label in labels:
        if label != guard and count_last == len(labels) - 1:
            assert len(set(labels[count_pre:count_last])) == 1
            count += count_last - count_pre + 1
            avg_feature = np.sum(features[count_pre:count_last], axis=0) / (count_last - count_pre)
            avg_feature = avg_feature.reshape(1, -1)
            avg_features.append(avg_feature)
            avg_feature = features[-1].reshape(1, -1)
            avg_features.append(avg_feature)
        elif count_last == len(labels) - 1:
            assert len(set(labels[count_pre:count_last + 1])) == 1
            count += count_last - count_pre + 1
            avg_feature = np.sum(features[count_pre:count_last + 1], axis=0) / (count_last - count_pre)
            avg_feature = avg_feature.reshape(1, -1)
            avg_features.append(avg_feature)
        elif label != guard:
            assert len(set(labels[count_pre:count_last])) == 1
            avg_feature = np.sum(features[count_pre:count_last], axis=0) / (count_last - count_pre)
            count += count_last - count_pre
            avg_feature = avg_feature.reshape(1, -1)
            avg_features.append(avg_feature)
            count_pre = count_last
            guard = label
        count_last += 1

    print(count)
    assert (count == len(labels))
    data_avg = preprocessing.normalize(np.squeeze(avg_features))
    np.save(save_path + f'{SD}_avg_feature.npy', data_avg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--SD', type=str,
                        help='Scenarios detail, For example: f512_r18_arc_class100, f128_r18_softmax_class100')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    main(parser.parse_args())
