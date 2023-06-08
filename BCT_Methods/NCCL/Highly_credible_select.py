import numpy as np
from sklearn import preprocessing
from scipy.spatial.distance import cdist

data = preprocessing.normalize(np.load('../feature_save/f128_softmax_70class_feature_total_not_nor.npy'))
print(data.shape)
labels = np.load('../feature_save/f128_softmax_70class_label_total_not_nor.npy')
print(labels.shape)
print('loaded!')

class_num = len(set(list(labels)))
U = np.log(class_num) / 2
print('U: ', U)

guard = 0
count_pre = 0
count_last = 0
Highly_credible_select_sig = []

for label in labels:
    if label != guard and count_last == len(labels) - 1:
        assert len(set(labels[count_pre:count_last])) == 1
        avg_feature = np.sum(data[count_pre:count_last], axis=0) / (count_last - count_pre)
        avg_feature = avg_feature.reshape(1, -1)
        dis = cdist(data[count_pre:count_last], avg_feature)
        dis_var = np.var(dis) + 1e-10
        tmp = -dis / dis_var
        pik1 = np.exp(tmp)
        pik2 = np.sum(pik1, axis=1, keepdims=True)
        pik = pik1 / (pik2 + 1e-10)
        Entropy = np.sum(-pik * np.log(pik + 1e-10), axis=1)
        Entropy_select = np.where(Entropy <= U, 1, 0).tolist()
        Highly_credible_select_sig += Entropy_select
        avg_feature = data[-1].reshape(1, -1)
        dis = cdist(data[count_pre:count_last], avg_feature)
        dis_var = np.var(dis) + 1e-10
        tmp = -dis / dis_var
        pik1 = np.exp(tmp)
        pik2 = np.sum(pik1, axis=1, keepdims=True)
        pik = pik1 / (pik2 + 1e-10)
        Entropy = np.sum(-pik * np.log(pik + 1e-10), axis=1)
        Entropy_select = np.where(Entropy <= U, 1, 0).tolist()
        Highly_credible_select_sig += Entropy_select

    elif count_last == len(labels) - 1:
        assert len(set(labels[count_pre:count_last + 1])) == 1
        avg_feature = np.sum(data[count_pre:count_last + 1], axis=0) / (count_last - count_pre)
        avg_feature = avg_feature.reshape(1, -1)
        dis = cdist(data[count_pre:count_last], avg_feature)
        dis_var = np.var(dis) + 1e-10
        tmp = -dis / dis_var
        pik1 = np.exp(tmp)
        pik2 = np.sum(pik1, axis=1, keepdims=True)
        pik = pik1 / (pik2 + 1e-10)
        Entropy = np.sum(-pik * np.log(pik + 1e-10), axis=1)
        Entropy_select = np.where(Entropy <= U, 1, 0).tolist()
        Highly_credible_select_sig += Entropy_select

    elif label != guard:
        assert len(set(labels[count_pre:count_last])) == 1
        avg_feature = np.sum(data[count_pre:count_last], axis=0) / (count_last - count_pre)
        avg_feature = avg_feature.reshape(1, -1)
        dis = cdist(data[count_pre:count_last], avg_feature)
        dis_var = np.var(dis) + 1e-10
        tmp = -dis / dis_var
        pik1 = np.exp(tmp)
        pik2 = np.sum(pik1, axis=1, keepdims=True)
        pik = pik1 / (pik2 + 1e-10)
        Entropy = np.sum(-pik * np.log(pik + 1e-10), axis=1)
        Entropy_select = np.where(Entropy <= U, 1, 0).tolist()
        Highly_credible_select_sig += Entropy_select
        count_pre = count_last
        guard = label
    count_last += 1

total_num = len(labels)
Highly_credible_num = sum(Highly_credible_select_sig)
print(f'Total number: {total_num} , Highly credible number: {Highly_credible_num}')

np.save('./feature_save/highly_credible_feature_sig.npy', np.array(Highly_credible_select_sig))
