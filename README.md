
# MixBCT: Towards Self-Adapting Backward-Compatible Training

## Introduction

Implementation of **[MixBCT: Towards Self-Adapting Backward-Compatible Training(Ours)](https://arxiv.org/abs/2308.06948)** , **L2** and other SOTA methods: [UniBCT](https://arxiv.org/abs/2203.01583), [NCCL](https://ojs.aaai.org/index.php/AAAI/article/view/20175), [BCT](http://openaccess.thecvf.com/content_CVPR_2020/html/Shen_Towards_Backward-Compatible_Representation_Learning_CVPR_2020_paper.html), [AdvBCT](https://openaccess.thecvf.com/content/CVPR2023/html/Pan_Boundary-Aware_Backward-Compatible_Representation_via_Adversarial_Learning_in_Image_Retrieval_CVPR_2023_paper.html)

**L2:**     Conduct simple L2 constraint between old features and new features

**BCT:**    [Towards Backward-Compatible Representation Learning](http://openaccess.thecvf.com/content_CVPR_2020/html/Shen_Towards_Backward-Compatible_Representation_Learning_CVPR_2020_paper.html) (CVPR2020) 

**UniBCT:** [Towards Universal Backward-Compatible Representation Learning](https://arxiv.org/abs/2203.01583) (AAAI2022)

**NCCL:**   [Neighborhood Consensus Contrastive Learning for Backward-Compatible Representation](https://ojs.aaai.org/index.php/AAAI/article/view/20175) (IJCAI2022)

**AdvBCT:** [Boundary-Aware Backward-Compatible Representation via Adversarial Learning in Image Retrieval](https://openaccess.thecvf.com/content/CVPR2023/html/Pan_Boundary-Aware_Backward-Compatible_Representation_via_Adversarial_Learning_in_Image_Retrieval_CVPR_2023_paper.html) (CVPR2023)

## Dataset

* **Training dataset: MS1M-V3** (ms1m-retinaface) ---- 5179510 images with 93431 IDs 
* **Eval dataset: IJB-C**
  
The download link of the datasets can be find in [https://github.com/deepinsight/insightface/tree/master/recognition/\_datasets\_]


## Code Structure

* The main-dir(./) is used for train the Old model
* ./BCT_methods/  --- The methods which summarized MixBCT, UniBCT, NCCL, BCT, AdvBCT and L2. 
* ./tools/        --- dataset split code, some preprocessing operations code and IJB-C evaluation code

This code based on the project [insightface](https://github.com/deepinsight/insightface), We maintain separate directories for each method to enhance clarity and facilitate reproducibility.


**Note**: We fixed the random seed in the main file for training, and this will significantly reduce the speed of training. You can speed up the training by comment out following two lines in the main file:
```
#torch.backends.cudnn.benchmark = False
#torch.backends.cudnn.deterministic = True
```
However, it will result in slight randomness of the results.


## Training Flow ---- An Example:

### Step-1
**Train the old model use the arcface loss.**
```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1  --master_port=22222 train_old_arc.py configs/f512_r18_arc_class30.py
```
**Or train the old model use the softmax loss.**
```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1  --master_port=22222 train_old_softmax.py configs/f128_r18_softmax_class30.py
```
### Step-2   ----（preprocessing operations）used in MixBCT、NCCL
**Get the feature of the dataset consist of 'class70'(the sub-dataset containing 70 percent of the classes) images.**  
```
python tools/get_feature/get_avg_feature.py configs/f128_r18_softmax_class30.py --SD f128_r18_softmax_class70
```
### Step-2   ----（preprocessing operations）used in BCT、UniBCT
**Get the avg feature of the dataset consist of 'class70' images(based on Step-2).**  
```
python tools/get_feature/get_avg_feature.py  --SD f128_r18_softmax_class70
```
### Step-2   ----（preprocessing operations）used in MixBCT
**Get the denoised feature of the dataset consist of 'class70' images(based on Step-2).** 
```
python tools/get_feature/denoise_credible.py --T 0.9 --SD f128_r18_softmax_class70
```
### Step-3  
**Train the New-Model by MixBCT**
```
cd BCT_Methods/MixBCT/
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1  --master_port=22222 train.py configs/OPclass_ms1mv3_r18_to_r50_MixBCT_softmax_to_arc_f128.py
```
**Or train the New-Model by NCCL**
```
cd BCT_Methods/NCCL/
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1  --master_port=22222 train.py configs/OPclass_ms1mv3_r18_to_r50_NCCL_softmax_to_arc_f128.py
```
**Or train the New model by other methods**
```
cd BCT_Methods/#Other Methods/
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1  --master_port=22222 train.py configs/OPclass_ms1mv3_r18_to_r50_(Othermethods)_softmax_to_arc_f128.py
```
### Step-4
**IJB-C evaluation**
```
**self-test 1:1**
python tools/ijbc_eval/ijbc_eval.py -m=#The path of 'New_model.pt' -net=#The backbone of Nld model(r18,r50,vit...) 
**self-test 1:N**
python tools/ijbc_eval/ijbc_eval.py -m=#The path of 'New_model.pt' -net=#The backbone of Nld model(r18,r50,vit...) -N 
**cross-test 1:1**
python tools/ijbc_eval/ijbc_eval.py -m=#The path of 'New_model.pt' -net=#The backbone of Nld model(r18,r50,vit...) -m_old=#The path of 'Old_model.pt' -old_net=#The backbone of Old model(r18,r50,vit...) 
**cross-test 1:N**
python tools/ijbc_eval/ijbc_eval.py -m=#The path of 'New_model.pt' -net=#The backbone of Nld model(r18,r50,vit...) -m_old=#The path of 'Old_model.pt' -old_net=#The backbone of Old model(r18,r50,vit...) 

# IJB-C evaluation-Using nohup
#CUDA_VISIBLE_DEVICES=0 nohup python tools/ijbc_eval/ijbc_eval.py -m=#The path of 'New_model.pt' -net=#The backbone of Nld model(r18,r50,vit...) >>#Result save path 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python tools/ijbc_eval/ijbc_eval.py -m=#The path of 'New_model.pt' -net=#The backbone of Nld model(r18,r50,vit...) -N >>#Result save path 2>&1 &
#CUDA_VISIBLE_DEVICES=2 nohup python tools/ijbc_eval/ijbc_eval.py -m=#The path of 'New_model.pt' -net=#The backbone of Nld model(r18,r50,vit...) -m_old=#The path of 'Old_model.pt' -old_net=#The backbone of Old model(r18,r50,vit...) >>#Result save path 2>&1 &
#CUDA_VISIBLE_DEVICES=3 nohup python tools/ijbc_eval/ijbc_eval.py -m=#The path of 'New_model.pt' -net=#The backbone of Nld model(r18,r50,vit...) -m_old=#The path of 'Old_model.pt' -old_net=#The backbone of Old model(r18,r50,vit...) >>#Result save path 2>&1 &
```

## Citation
If this repository helps your research, please cite our paper:

```
@article{liang2023mixbct,
  title={MixBCT: Towards Self-Adapting Backward-Compatible Training},
  author={Liang, Yu and Zhang, Shiliang and Wang, Yaowei and Xiao, Sheng and Li, Kenli and Wang, Xiaoyu},
  journal={arXiv preprint arXiv:2308.06948},
  year={2023}
}
```
