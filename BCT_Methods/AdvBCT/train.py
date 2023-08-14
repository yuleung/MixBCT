import argparse
import logging
import os

import numpy as np
import torch
from typing import List
from torch import distributed
from torch.utils.tensorboard import SummaryWriter

from backbones import get_model_advBCT
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
import json
from extract_feat import gen_class_meta





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


def load_old_meta(config):
    print(f'start to load metadata from {config.center_radius}')
    path = config.center_radius
    if not os.path.exists(path):
        print(f'{config.center_radius} doesnt exist, which is required by some models')
        return None
    else:
        with open(path, 'r') as f:
            old_meta = json.load(f)
        return old_meta





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
     
     
    gen_class_meta(feat=cfg.old_embedding, label=cfg.old_embedding_label, name=cfg.center_radius)
    old_feature = F.normalize(torch.tensor(np.array(np.load(cfg.old_embedding))))


    train_loader = get_dataloader(
        cfg.rec, local_rank=args.local_rank, batch_size=cfg.batch_size, dali=cfg.dali)
    backbone = get_model_advBCT(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size,num_class=cfg.num_classes
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
    start_epoch = 0
    global_step = 0
    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint




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
    #start_epoch = 0
    #global_step = 0
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    old_meta = load_old_meta(cfg)
    model_criterion = torch.nn.NLLLoss()
    for epoch in range(start_epoch, cfg.num_epoch):
        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        for batch_idx, (img, local_labels, index) in enumerate(train_loader):
            global_step += 1

            #get alpha for loss L_adv
            p = float(batch_idx + epoch * len(train_loader)) / cfg.num_epoch / len(train_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            #get radius_eb for loss L_p2s
            radius_eb = torch.zeros(cfg.num_classes)
            for j in range(len(local_labels)):
                if str(local_labels[j].item()) in old_meta:
                    radius_max = old_meta[str(local_labels[j].item())]['radius'][-1]
                    radius_eb[local_labels[j].item()] = abs(radius_max - cfg.threshold)
                else:
                    radius_eb[local_labels[j].item()] = 0

            #get old embeddings
            index = index.detach().cpu()
            old_local_embeddings = old_feature[index]

            #New model forward
            local_embeddings, model_out_new, model_out_old, radius_eb = backbone(img, old_local_embeddings, alpha, radius_eb)

            #calculate the loss of L_p2s

            Loss_p2s = 0.
            feat = F.normalize(local_embeddings)
            count = 0
            for j in range(len(feat)):
                if str(local_labels[j].item()) in old_meta:
                    diff = feat[j] - torch.tensor(old_meta[str(local_labels[j].item())]['center'])[None, :].to(
                        feat.device)
                    if 1:
                        if old_meta[str(local_labels[j].item())]['radius'][-1] < cfg.threshold:
                            radius = old_meta[str(local_labels[j].item())]['radius'][-1] + radius_eb[
                                local_labels[j].item()]
                            if batch_idx % cfg.frequent  == 0 and j == 0:
                                print(f'original {old_meta[str(local_labels[j].item())]["radius"][-1]},+ {radius_eb[local_labels[j].item()].item()}, radius {radius.item()}')
                        else:
                            radius = old_meta[str(local_labels[j].item())]['radius'][-1] - radius_eb[
                                local_labels[j].item()]
                            if batch_idx % cfg.frequent  == 0 and j == 0:
                                print(f'original {old_meta[str(local_labels[j].item())]["radius"][-1]},- {radius_eb[local_labels[j].item()].item()}, radius {radius.item()}')
                    if len(old_meta[str(local_labels[j].item())]['radius']) <= 1:
                        continue
                    tmp = max(torch.norm(diff, p=2) - radius, 0)
                    if tmp > 0.:
                        Loss_p2s += tmp
                        count += 1
            if count:
                Loss_p2s /= count
                Loss_p2s *= 4

            #calculate the Loss of L_adv
            model_label_new = torch.zeros(len(local_labels)).long().cuda()
            model_label_old = torch.ones(len(local_labels)).long().cuda()
            Loss_adv = model_criterion(model_out_new, model_label_new) + model_criterion(model_out_old, model_label_old)

            #calculate the loss of L_cls
            Loss_cls = module_partial_fc(local_embeddings, local_labels, opt)
            Loss_adv = 4 * Loss_adv * (cfg.num_epoch - epoch + 1)/ cfg.num_epoch
            loss = Loss_cls + Loss_p2s +  Loss_adv

            if global_step % 50 == 0 and global_step > 0 and rank == 0:
                print(f'Loss_cls: {round(Loss_cls.item(), 4)}, Loss_p2s:  {round(Loss_p2s.item(), 4)}, Loss_adv:{round(Loss_adv.item(), 4)}, Total_loss: {round(loss.item(), 4)}')

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

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))

        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            torch.save(backbone.module.state_dict(), path_module)


        '''
        path_pfc = os.path.join(cfg.output, "softmax_fc_gpu_{}.pt".format(rank))
        torch.save(module_partial_fc.state_dict(), path_pfc)
        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            torch.save(backbone.module.state_dict(), path_module)
        '''
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
