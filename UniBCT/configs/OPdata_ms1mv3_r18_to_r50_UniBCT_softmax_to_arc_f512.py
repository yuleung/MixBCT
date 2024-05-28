from easydict import EasyDict as edict

config = edict()
config.loss = "arcface"
config.network = "r50"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.1

config.old_model_path = '../../work_dirs/f512_r18_softmax_data30/model.pt'
config.old_prototype = '../../feature_save/OPdata/f512_r18_softmax_data70_avg_feature.npy'

config.rec_val = '../../dataset/ms1m-retinaface-t1'
config.rec = '../../dataset/ms1m-retinaface-t1/train_part_70_data'
config.num_classes = 93431
config.num_image = 5179510-1554138
config.num_epoch = 35
config.warmup_epoch = -1
config.decay_epoch = [20, 26, 32]
config.val_targets =[]