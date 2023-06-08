import argparse
import logging
import os

import numpy as np
import torch
from torch import distributed
from torch.utils.tensorboard import SummaryWriter

from backbones import get_model
from dataset import get_dataloader
from torch.utils.data import DataLoader
from lr_scheduler import PolyScheduler
from losses import CosFace, ArcFace
from partial_fc import PartialFC, PartialFCAdamW
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging
from sklearn import preprocessing
from compatible_learning_loss import CompatibleLearningLoss

try:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    distributed.init_process_group("nccl")
except KeyError:
    world_size = 1
    rank = 0
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
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

    torch.cuda.set_device(args.local_rank)
    cfg = get_config(args.config)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )
    train_loader = get_dataloader(
        cfg.rec, local_rank=args.local_rank, batch_size=cfg.batch_size, dali=cfg.dali)
    backbone = get_model(
        cfg.network, num_features=cfg.embedding_size, dropout=0.0, fp16=cfg.fp16).cuda()
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)

    margin_loss = ArcFace()

    if rank == 0:
        print(backbone)

    if cfg.optimizer == "sgd":
        module_partial_fc = PartialFC(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, cfg.fp16)
        module_partial_fc.train().cuda()
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFCAdamW(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, cfg.fp16)
        module_partial_fc.train().cuda()
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch
    lr_scheduler = PolyScheduler(
        optimizer=opt,
        base_lr=cfg.lr,
        max_steps=cfg.total_step,
        warmup_steps=cfg.warmup_step
    )

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec_val, summary_writer=summary_writer
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        writer=summary_writer
    )
    '''
    'Highly_credible_select.py' is used for detect the noise in the old feature.
    However, the detection results of the method for detecting noise in NCCL 
    on the ms1mv3 dataset determined that almost all samples were trustworthy samples,
    we directly used all the old features for simplicity.
    '''
    old_feature_highly_credible = torch.tensor(preprocessing.normalize(np.load(cfg.old_embedding))).cuda()
    # old_feature_highly_credible = torch.tensor(np.load('././feature_save/highly_credible_feature_sig.npy'))

    cl_loss = CompatibleLearningLoss(cfg.embedding_size, cfg.num_classes, cfg.queue_size).cuda()
    alpha = cfg.alpha
    beta = cfg.beta

    loss_am = AverageMeter()
    start_epoch = 0
    global_step = 0
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        for _, (img, labels, index) in enumerate(train_loader):
            global_step += 1
            old_embeddings = old_feature_highly_credible[index].detach()
            new_embeddings = backbone(img)
            old_logits, old_embeddings, _ = module_partial_fc(old_embeddings, labels, opt, return_loss=False)
            old_logits = old_logits.detach()
            new_logits, new_embeddings, labels, onehot_loss = module_partial_fc(new_embeddings, labels, opt, return_loss=True)
            l1_loss, l2_loss  = cl_loss(old_embeddings, old_logits, new_embeddings, new_logits, labels)
            loss = onehot_loss + alpha * l1_loss + beta * l2_loss

            if rank == 0 and global_step > 0 and global_step % 20 == 0:
                print('onehot_loss:{:.3f} l1_loss:{:.3f} l2_loss:{:.3f} total_loss:{:.3f}'.format(
                    onehot_loss.item(), alpha*l1_loss.item(), beta*l2_loss.item(), loss.item()))

            if cfg.fp16:
                amp.scale(loss).backward()
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                amp.step(opt)
                amp.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                opt.step()

            opt.zero_grad()
            lr_scheduler.step()

            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)

                if global_step % cfg.verbose == 0 and global_step > 200:
                    callback_verification(global_step, backbone)

        path_pfc = os.path.join(cfg.output, "softmax_fc_gpu_{}.pt".format(rank))
        torch.save(module_partial_fc.state_dict(), path_pfc)
        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            torch.save(backbone.module.state_dict(), path_module)

        if cfg.dali:
            train_loader.reset()

    path_pfc = os.path.join(cfg.output, "softmax_fc_gpu_{}.pt".format(rank))
    torch.save(module_partial_fc.state_dict(), path_pfc)
    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)

    distributed.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    main(parser.parse_args())
