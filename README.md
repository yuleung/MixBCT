# MixBCT: Towards Self-Adapting Backward Compatible Training
Implementation of MixBCT and other SOTA mathods: UniBCT, NCCL, BCT


* The main dir is used for train the Old model
* BCT_methods/  --- summarized MixBCT, UniBCT, NCCL, BCT
* tools/        --- some preprocessing operations and ijb-c evaluation code


### Training Flow ---- An Example:
```
## Step-1
# Train the Old model use the arcface loss
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1  --master_port=22222 train_old_arc.py configs/f512_r18_arc_class30.py
# OR Train the Old model use the softmax loss
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1  --master_port=22222 train_old_softmax.py configs/f128_r18_softmax_class30.py

## Step-2
# Get the feature of the dataset consist of 'class70' images.                   ----used in MixBCT、NCCL
python tools/get_feature/get_avg_feature.py configs/f128_r18_softmax_class30.py --SD f128_r18_softmax_class70

## Step-3
# Get the avg feature of the dataset consist of 'class70' images.               ----used in BCT、UniBCT
python tools/get_feature/get_avg_feature.py  --SD f128_r18_softmax_class70

## Step-4
# Get the denoised feature of the dataset consist of 'class70' images(based on Step-2).          ----used in MixBCT
python tools/get_feature/denoise_credible.py --T 0.9 --SD f128_r18_softmax_class70

## Step-5
# Train the New model by MixBCT
cd BCT_Methods/MixBCT/
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1  --master_port=22222 train.py configs/OPclass_ms1mv3_r18_to_r50_MixBCT_softmax_to_arc_f128.py
# OR Train the New model by NCCL
cd BCT_Methods/NCCL/
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1  --master_port=22222 train.py configs/OPclass_ms1mv3_r18_to_r50_NCCL_softmax_to_arc_f128.py
# OR Train the New model by other methods
cd BCT_Methods/#Other methods/
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1  --master_port=22222 train.py configs/OPclass_ms1mv3_r18_to_r50_(Othermethods)_softmax_to_arc_f128.py

## Step-6
# IJB-C evaluation
# self-test 1:1
python tools/ijbc_eval/ijbc_eval.py -m=#The path of 'New_model.pt' -net=#The backbone of Nld model(r18,r50,vit...) 
# self-test 1:N
python tools/ijbc_eval/ijbc_eval.py -m=#The path of 'New_model.pt' -net=#The backbone of Nld model(r18,r50,vit...) -N 
# cross-test 1:1
python tools/ijbc_eval/ijbc_eval.py -m=#The path of 'New_model.pt' -net=#The backbone of Nld model(r18,r50,vit...) -m_old=#The path of 'Old_model.pt' -old_net=#The backbone of Old model(r18,r50,vit...) 
# cross-test 1:N
python tools/ijbc_eval/ijbc_eval.py -m=#The path of 'New_model.pt' -net=#The backbone of Nld model(r18,r50,vit...) -m_old=#The path of 'Old_model.pt' -old_net=#The backbone of Old model(r18,r50,vit...) 

(
# IJB-C evaluation using nohup
#CUDA_VISIBLE_DEVICES=0 nohup python tools/ijbc_eval/ijbc_eval.py -m=#The path of 'New_model.pt' -net=#The backbone of Nld model(r18,r50,vit...) >>#Result save path 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python tools/ijbc_eval/ijbc_eval.py -m=#The path of 'New_model.pt' -net=#The backbone of Nld model(r18,r50,vit...) -N >>#Result save path 2>&1 &
#CUDA_VISIBLE_DEVICES=2 nohup python tools/ijbc_eval/ijbc_eval.py -m=#The path of 'New_model.pt' -net=#The backbone of Nld model(r18,r50,vit...) -m_old=#The path of 'Old_model.pt' -old_net=#The backbone of Old model(r18,r50,vit...) >>#Result save path 2>&1 &
#CUDA_VISIBLE_DEVICES=3 nohup python tools/ijbc_eval/ijbc_eval.py -m=The path of 'New_model.pt' -net=#The backbone of Nld model(r18,r50,vit...) -m_old=#The path of 'Old_model.pt' -old_net=#The backbone of Old model(r18,r50,vit...) >>#Result save path 2>&1 &
)
 
```
