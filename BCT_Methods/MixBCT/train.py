import argparse
import logging
import os

import numpy as np
import torch
from typing import List
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
import torch.nn.functional as F
import random

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
    torch.cuda.manual_seed(seed)
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

    old_feature = F.normalize(torch.tensor(np.array(np.load(cfg.old_embedding))))
    credible_label = torch.tensor(np.load(cfg.credible_old_feature)).to(args.local_rank)

    train_loader = get_dataloader(
        cfg.rec, local_rank=args.local_rank, batch_size=cfg.batch_size, dali=cfg.dali)
    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size
    ).cuda()
    if rank == 0:
        print(backbone)
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16)

    margin_loss = ArcFace()

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

    loss_am = AverageMeter()
    start_epoch = 0
    global_step = 0
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    for epoch in range(start_epoch, cfg.num_epoch):
        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        for _, (img, local_labels, index) in enumerate(train_loader):
            global_step += 1

            local_embeddings = backbone(img)
            index = index.detach().cpu()
            old_local_embeddings = old_feature[index]
            old_credible_label = credible_label[index]
            '''
            with torch.no_grad():
                if _ == 0 and rank == 0:
                    loss_new = module_partial_fc(local_embeddings, local_labels, opt)
                    loss_old = module_partial_fc(old_local_embeddings.cuda(), local_labels, opt)
                    print('new CE loss: ', loss_new.item())
                    print('old CE loss: ', loss_old.item())
                    guard = 1
            '''
            credible_index = torch.where(old_credible_label == 1)[0]
            random_replace = random.sample(range(0, len(credible_index)), int(cfg.batch_size * cfg.mix_ratio))
            random_replace = credible_index[random_replace].cuda()
            local_embeddings[random_replace] = old_local_embeddings[random_replace].cuda()

            loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels, opt)

            if global_step % 50 == 0 and global_step > 0 and rank == 0:
                print('Mixed CE loss: ', loss.item())

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
