import sys
import os

print(os.path.abspath('./'))
sys.path.append('./')

import argparse
import logging
import numpy as np
import torch
from typing import List
from torch import distributed
from torch.utils.tensorboard import SummaryWriter
from backbones import get_model
from dataset import get_dataloader
from torch.utils.data import DataLoader
from lr_scheduler import PolyScheduler
from losses import CombinedMarginLoss, CosFace, ArcFace
from partial_fc import PartialFC, PartialFCAdamW
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging

try:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    distributed.init_process_group("nccl")
except KeyError:
    world_size = 1
    rank = 0
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12570",
        rank=rank,
        world_size=world_size,
    )


def main(args):
    seed = 666
    seed = seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    SD = args.SD
    torch.cuda.set_device(args.local_rank)
    cfg = get_config(args.config)
    os.makedirs(cfg.output, exist_ok=True)
    bs = 256
    init_logging(rank, cfg.output)

    data_name = cfg.rec

    SD_tail = SD.split('.')[0].split('_')[-1]
    save_path = './feature_save/'
    if SD_tail == 'class100':
        data_name = data_name.split('_')[0]
        save_path = save_path + 'EXclass/'
    elif SD_tail == 'data100':
        data_name = data_name.split('_')[0]
        save_path = save_path + 'EXdata/'
    elif SD_tail == 'class70':
        data_name = data_name.split('_')[0] + '_part_70_class'
        save_path = save_path + 'OPclass/'
    elif SD_tail == 'data70':
        data_name = data_name.split('_')[0] + '_part_70_data'
        save_path = save_path + 'OPdata/'
    elif SD_tail == 'class30':
        data_name = data_name.split('_')[0] + '_part_30_class'
        save_path = save_path + 'OPclass/'
    elif SD_tail == 'data30':
        data_name = data_name.split('_')[0] + '_part_30_data'
        save_path = save_path + 'OPdata/'

    print('Get feature of dataset: ', data_name)
    print('Save to: ', save_path)

    train_loader = get_dataloader(data_name, local_rank=args.local_rank, batch_size=bs, dali=cfg.dali, shuffle=False,
                                  drop_last=False, num_workers=4)
    backbone = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()
    # print(cfg.output)
    backbone_pth = os.path.join(cfg.output, "model.pt")
    new_state_dict = {k: v for k, v in torch.load(backbone_pth, map_location=torch.device(args.local_rank)).items() if
                      k != 'cls_fc.bias' and k != 'cls_fc.weight'}
    backbone.load_state_dict(new_state_dict, strict=True)

    if rank == 0:
        logging.info("backbone resume successfully!")
    backbone.eval()
    epoch_feature = []
    epoch_label = []
    with torch.no_grad():
        for step, (img, label, index) in enumerate(train_loader):
            features = backbone(img)
            feature = features.detach().cpu().numpy()
            epoch_feature.append(feature)
            epoch_label.append(label.detach().cpu().numpy())
            if (step + 1) % 100 == 0:
                print('samples:', (step + 1) * bs)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    feature_total = np.array(list(np.vstack(np.array(epoch_feature[:-1]))) + list(np.array(epoch_feature[-1])))
    np.save(save_path + f'{SD}_feature.npy', feature_total)
    label_total = np.array(list(np.hstack(np.array(epoch_label[:-1]))) + list(np.array(epoch_label[-1])))
    np.save(save_path + f'{SD}_label.npy', label_total)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('config', type=str, help='py config file')
    parser.add_argument('--SD', type=str,
                        help='Scenarios detail, For example: f512_r18_arc_class100, f128_r18_softmax_class100')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    main(parser.parse_args())
