import argparse
import logging
import os

import numpy as np
import torch
from typing import List
from torch import distributed
from torch.utils.tensorboard import SummaryWriter

from backbones import get_model_bct, get_model

from dataset import get_dataloader, get_dataloader_noddp
from torch.utils.data import DataLoader
from lr_scheduler import PolyScheduler
from losses import CosFace, ArcFace
from partial_fc import PartialFC, PartialFCAdamW
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging
import torch.nn.functional as F
from refine_prototype import PrototypeGeneration

# import collections
# from collections import OrderedDict

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
    train_loader_for_Proto = get_dataloader_noddp(
        cfg.rec, local_rank=args.local_rank, batch_size=128, dali=cfg.dali, shuffle=False, drop_last=False)
    backbone = get_model_bct(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size,
        syn_weight=torch.tensor(np.load(cfg.old_prototype))
    ).cuda()


    margin_loss = ArcFace()


    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)

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

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=1, gamma=0.1)
    '''
    lr_scheduler = PolyScheduler(
        optimizer=opt,
        base_lr=cfg.lr,
        max_steps=cfg.total_step,
        warmup_steps=cfg.warmup_step
    )
    '''
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

    global_step = 0
    start_epoch = 0

    loss_am = AverageMeter()


    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    for epoch in range(start_epoch, cfg.num_epoch):

        if epoch in [10, 20]:
            with torch.no_grad():
                old_model = get_model('r18', dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size)
                backbone_pth = cfg.old_model_path
                new_state_dict = {k: v for k, v in torch.load(backbone_pth, map_location=torch.device('cpu')).items() if
                                  k not in ["cls_fc.weight", "cls_fc.bias", "NormFace.param.0", "NormFace.param.1"]}
                old_model.load_state_dict(new_state_dict, strict=True)
                PrototypeGeneration(cfg, backbone, old_model, loader=train_loader_for_Proto, rank=rank)
                del old_model
                torch.cuda.empty_cache()
                torch.distributed.barrier()
                w_o_head = np.load(cfg.output + '/old_prototype.npy')
                for name, param in backbone.named_parameters():
                    if name == 'module.BCT_FC.weight':
                        print(param[0])
                        param[:] = torch.tensor(w_o_head).transpose(0, 1)
                for name, param in backbone.named_parameters():
                    if name == 'module.BCT_FC.weight':
                        print(param[0])
                print("loaded old module_fc !")
                backbone.train()


        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        for _, (img, local_labels, _) in enumerate(train_loader):
            global_step += 1
            local_embeddings, bct_out = backbone(img)
            onehot_loss = module_partial_fc(local_embeddings, local_labels, opt)
            BCT_loss = 1 * F.cross_entropy(bct_out, local_labels)
            if epoch < 10:
                loss = onehot_loss + 0 * BCT_loss
            else:
                loss = onehot_loss + 1 * BCT_loss

            if rank == 0 and global_step > 0 and global_step % 50 == 0:
                print('onehot_loss: ', onehot_loss.item())
                print('BCT_loss: ', BCT_loss.item())
                print('total_loss: ', loss.item())

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


            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)

                if global_step % cfg.verbose == 0 and global_step > 200:
                    callback_verification(global_step, backbone)

        if epoch + 1 in cfg.decay_epoch:
            lr_scheduler.step()

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
