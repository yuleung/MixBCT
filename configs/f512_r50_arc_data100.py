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
config.dali = False
config.lr = 0.1
config.rec_val = './dataset/ms1m-retinaface-t1'
config.rec = './dataset/ms1m-retinaface-t1/train'
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 35
config.warmup_epoch = 0
config.val_targets = []#["lfw", "cfp_fp", "agedb_30"]
