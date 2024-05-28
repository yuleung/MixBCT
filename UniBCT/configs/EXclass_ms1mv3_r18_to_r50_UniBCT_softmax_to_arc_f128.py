from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.loss = "arcface"
config.network = "r50"
config.resume = False
config.output = None
config.embedding_size = 128
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.1

config.old_model_path = '../../work_dirs/f128_r18_softmax_class30/model.pt'
config.old_prototype = '../../feature_save/EXclass/f128_r18_softmax_class100_avg_feature.npy'

config.rec_val = '../../dataset/ms1m-retinaface-t1'
config.rec = '../../dataset/ms1m-retinaface-t1/train'
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 35
config.warmup_epoch = -1
config.decay_epoch = [20, 26, 32]
config.val_targets = []#["lfw", "cfp_fp", "agedb_30"]
