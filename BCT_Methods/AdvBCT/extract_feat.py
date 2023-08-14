
import torch
import os
import argparse
import json
import numpy as np
import math
from collections import defaultdict
from sklearn import preprocessing


def gen_class_meta(feat,label, name, save_path):
    #feat = np.load(os.path.join(config.EVAL.SAVE_DIR, f'{config.TRAIN.DATASET}_{feat_num}_feat.npy'))
    #label = np.load(os.path.join(config.EVAL.SAVE_DIR, f'{config.TRAIN.DATASET}_{feat_num}_label.npy'))
    print('feat, label', feat.shape, label.shape)
    feat_dict = defaultdict(list)
    feat = preprocessing.normalize(feat)
    for i in range(len(label)):
        #feat[i] = feat[i] / np.linalg.norm(feat[i])
        feat_dict[int(label[i])].append(feat[i])

    data_dict = {}
    for i, k in enumerate(feat_dict.keys()):
        center = np.asarray(feat_dict[k]).mean(0)
        # center = center/np.linalg.norm(center)
        # print('ddd', np.asarray(feat_dict[k]).shape, center.shape)
        maxtmp = 0
        radius = []
        for f in feat_dict[k]:
            # f = f/np.linalg.norm(f)
            # diff = f - center/np.linalg.norm(center)
            diff = f - center
            tmp = np.linalg.norm(diff)
            maxtmp = max(tmp, maxtmp)
            radius.append(tmp.item())

        radius = sorted(radius)
        # 1.5IQR
        radius_new = []
        maxtmp = 0
        if len(radius) >= 4:
            nu = len(radius)
            q3, q1 = radius[int(3 * nu / 4)], radius[int(nu / 4)]
            IQR = q3 - q1
            for r in radius:
                if r < q1 - 1.5 * IQR or r > q3 + 1.5 * IQR:
                    continue
                maxtmp = max(r, maxtmp)
                radius_new.append(r)
        else:
            for r in radius:
                maxtmp = max(r, maxtmp)
                radius_new.append(r)
        if len(radius_new) == 0:
            radius_new = radius

        data_dict[k] = {'center': center.tolist(), 'radius': radius_new}
        #print(i, len(feat_dict), k, len(radius), radius, maxtmp)
    # break
    with open(os.path.join('./feature_save', name), 'w') as fw:
        json.dump(data_dict, fw)
'''
def gen_class_theta(feat,label, name):
    #feat = np.load(os.path.join(config.EVAL.SAVE_DIR, f'{config.TRAIN.DATASET}_{feat_num}_feat.npy'))
    #label = np.load(os.path.join(config.EVAL.SAVE_DIR, f'{config.TRAIN.DATASET}_{feat_num}_label.npy'))
    print('feat, label', feat.shape, label.shape)
    feat_dict = defaultdict(list)
    for i in range(len(label)):
        feat[i] = feat[i] / np.linalg.norm(feat[i])
        feat_dict[int(label[i])].append(feat[i])

    data_dict = {}
    for i, k in enumerate(feat_dict.keys()):
        center = np.asarray(feat_dict[k]).mean(0)
        # center = center/np.linalg.norm(center)
        # print('ddd', np.asarray(feat_dict[k]).shape, center.shape)
        radius = []
        for f in feat_dict[k]:
            # f = f / np.linalg.norm(f)
            diff = min(np.dot(f,center/np.linalg.norm(center)),1)
            theta = math.acos(diff)
            radius.append(theta)
        radius = sorted(radius)
        #1.5IQR
        radius_new = []
        maxtmp = 0
        if len(radius)>=4:
            nu = len(radius)
            q3,q1 = radius[int(3*nu/4)], radius[int(nu/4)]
            IQR = q3-q1
            for r in radius:
                if r < q1-1.5*IQR or r > q3+1.5*IQR:
                    continue
                maxtmp = max(r, maxtmp)
                radius_new.append(r)
        else:
            for r in radius:
                maxtmp = max(r, maxtmp)
                radius_new.append(r)
        if len(radius_new) == 0:
            radius_new = radius

        data_dict[k] = {'center': center.tolist(), 'radius': radius_new}
        #print(i, len(feat_dict), k, len(radius_new), radius_new, maxtmp)
    # break
    with open(os.path.join('./feature_save', f'{name}_meta_theta_centernorm_after1.5iqr.json'), 'w') as fw:    #the save path
        json.dump(data_dict, fw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('BCT training script', add_help=False)
    #parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args, _ = parser.parse_known_args()
    feature = np.load('')   #The path of the training features produced by the old model   Example:  feature_save/EXdata/f512_softmax_data70_feature.npy')
    label = np.load('')     #The path of the training labels produced by the old model   Example:  feature_save/EXdata/f512_softmax_data70_label.npy')
    name = ''               #The prefix of the result    Example: 'f512_softmax_70data'
    gen_class_meta(feature,label, name)   #the default setting in the original paper
    #gen_class_theta(feature,label, name, save_path='./feature_save')
'''