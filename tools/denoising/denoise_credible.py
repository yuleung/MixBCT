import numpy as np
from sklearn import preprocessing
from scipy.spatial.distance import cdist
import math
import argparse
import sys
import os

sys.path.append('./')

def process_chunk(data_chunk, T):
    if len(data_chunk) == 0:
        return []
    
    avg_feature = np.mean(data_chunk, axis=0).reshape(1, -1)
    
    distances = cdist(data_chunk, avg_feature).ravel()
    sorted_indices = np.argsort(distances)
    
    keep_threshold = math.ceil(T * len(data_chunk))
    true_samples = sorted_indices[:keep_threshold]
    
    mask = np.zeros(len(data_chunk), dtype=bool)
    mask[true_samples] = True
    return mask.tolist()

def main(args):
    SD = args.SD
    
    SD_tail = SD.split('_')[-1]
    save_path = './feature_save/'
    if SD_tail in ['class100', 'data100']:
        save_path = os.path.join(save_path, 'EXclass/' if 'class' in SD_tail else 'EXdata/')
    else:
        save_path = os.path.join(save_path, 'OPclass/' if 'class' in SD_tail else 'OPdata/')

    data = np.load(os.path.join(save_path, f'{SD}_feature.npy'))
    labels = np.load(os.path.join(save_path, f'{SD}_label.npy'))

    data = preprocessing.normalize(data)    
    data = preprocessing.normalize(data, axis=0) 

    credible_label = []
    current_start = 0
    current_label = labels[0]

    for i in range(1, len(labels)):
        if labels[i] != current_label:
            chunk = data[current_start:i]
            chunk_labels = process_chunk(chunk, args.T)
            credible_label.extend(chunk_labels)
            
            current_start = i
            current_label = labels[i]

    final_chunk = data[current_start:]
    final_labels = process_chunk(final_chunk, args.T)
    credible_label.extend(final_labels)

    assert len(credible_label) == len(labels), f"Length mismatch: {len(credible_label)} vs {len(labels)}"

    np.save(os.path.join(save_path, f'{SD}_credible_ratio_{int(args.T*100)}_feature.npy'), credible_label)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Denoising and get credible samples.')
    parser.add_argument('--T', type=float, default=0.9, help='threthold')
    parser.add_argument('--SD', type=str, help='Scenarios detail, For example: f512_r18_arc_class100, f128_r18_softmax_class100')
    args = parser.parse_args()


    if not 0 <= args.T <= 1:
        raise ValueError("Threshold T must be in [0, 1]")
    
    main(args)
