#Implementation of MixBCT: Towards Self-Adapting Backward-Compatible Training(Ours) , L2 and other SOTA methods: UniBCT, NCCL, BCT, AdvBCT

#L2: Conduct simple L2 constraint between old features and new features

#BCT: Towards Backward-Compatible Representation Learning (CVPR2020)

#UniBCT: Towards Universal Backward-Compatible Representation Learning (AAAI2022)

#NCCL: Neighborhood Consensus Contrastive Learning for Backward-Compatible Representation (IJCAI2022)

#AdvBCT: Boundary-Aware Backward-Compatible Representation via Adversarial Learning in Image Retrieval (CVPR2023)



A Example:

#Train the Old model use the arcface loss
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1  --master_port=22222 train_old_arc.py configs/f512_r18_arc_class30.py
#Train the Old model use the softmax loss
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1  --master_port=22222 train_old_softmax.py configs/f128_r18_softmax_class30.py

#Get the feature of the dataset has 'class70' images.                   ----used in MixBCT,NCCL,AdvBCT
python tools/get_feature/get_feature.py configs/f128_r18_softmax_class30.py --SD f128_r18_softmax_class70

#Get the avg feature of the dataset has 'class70' images.               ----used in BCT,UniBCT
python tools/get_feature/get_avg_feature.py  --SD f128_r18_softmax_class70
 
#Get the denoised feature of the dataset has 'class70' images.          ----used in MixBCT
python tools/get_feature/denoise_credible.py --T 0.9 --SD f128_r18_softmax_class30

#Train the New model by MixBCT
cd BCT_Methods/MixBCT/
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1  --master_port=22222 train.py configs/OPclass_ms1mv3_r18_to_r50_MixBCT_softmax_to_arc_f128.py


# IJB-C evaluation
# self-test 1:1
python tools/ijbc_eval/ijbc_eval.py -m=The path of 'New_model.pt' -net=The backbone of Nld model(r18,r50,vit...) 
# self-test 1:N
python tools/ijbc_eval/ijbc_eval.py -m=The path of 'New_model.pt' -net=The backbone of Nld model(r18,r50,vit...) -N 
# cross-test 1:1
python tools/ijbc_eval/ijbc_eval.py -m=The path of 'New_model.pt' -net=The backbone of Nld model(r18,r50,vit...) -m_old=#The path of 'Old_model.pt' -old_net=The backbone of Old model(r18,r50,vit...) 
# cross-test 1:N
python tools/ijbc_eval/ijbc_eval.py -m=The path of 'New_model.pt' -net=The backbone of Nld model(r18,r50,vit...) -m_old=#The path of 'Old_model.pt' -old_net=The backbone of Old model(r18,r50,vit...) 

#CUDA_VISIBLE_DEVICES=0 nohup python tools/ijbc_eval/ijbc_eval.py -m=The path of 'New_model.pt' -net=The backbone of Nld model(r18,r50,vit...) >>Result save path 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python tools/ijbc_eval/ijbc_eval.py -m=The path of 'New_model.pt' -net=The backbone of Nld model(r18,r50,vit...) -N >>Result save path 2>&1 &
#CUDA_VISIBLE_DEVICES=2 nohup python tools/ijbc_eval/ijbc_eval.py -m=The path of 'New_model.pt' -net=The backbone of Nld model(r18,r50,vit...) -m_old=#The path of 'Old_model.pt' -old_net=The backbone of Old model(r18,r50,vit...) >>Result save path 2>&1 &
#CUDA_VISIBLE_DEVICES=3 nohup python tools/ijbc_eval/ijbc_eval.py -m=The path of 'New_model.pt' -net=The backbone of Nld model(r18,r50,vit...) -m_old=#The path of 'Old_model.pt' -old_net=The backbone of Old model(r18,r50,vit...) >>Result save path 2>&1 &
