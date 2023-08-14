from easydict import EasyDict as edict

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
config.dali = False
config.lr = 0.1

config.old_embedding = '../../feature_save/OPclass/f128_r18_softmax_class70_feature.npy'
config.center_radius = '../../feature_save/OPclass/f128_r18_softmax_class70_meta_radius_centernorm_for_AdvBCT.npy'
config.old_embedding_label = '../../feature_save/OPclass/f128_r18_softmax_class70_label.npy'

config.rec_val = '../../dataset/ms1m-retinaface-t1'
config.rec = '../../dataset/ms1m-retinaface-t1/train_part_70_class'
config.num_classes = 93431 - 28029
config.num_image = 5179510 - 1581241
config.num_epoch = 35
config.warmup_epoch = 0
config.val_targets = [] #["lfw", "cfp_fp", "agedb_30"]
