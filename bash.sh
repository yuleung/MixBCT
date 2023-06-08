

python tools/get_feature/get_avg_feature.py configs/f128_r18_softmax_class30.py --SD f128_r18_softmax_class30

python tools/get_feature/get_avg_feature.py  --SD f128_r18_softmax_class30

python tools/get_feature/denoise_credible.py --T 0.9 --SD f128_r18_softmax_class30




CUDA_VISIBLE_DEVICES=0 nohup python ijb_evals2_vit_f128.py -m=work_dirs/softmax_ms1mv3_r18_per70class_nccl_128/model.pt -net=r18 >> ./result_log/softmax_ms1mv3_r18_per70class_nccl_128_selftest_11.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python ijb_evals2_vit_f128.py -m=work_dirs/softmax_ms1mv3_r18_per70class_nccl_128/model.pt -net=r18 -N >> ./result_log/softmax_ms1mv3_r18_per70class_nccl_128_selftest_1N.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python ijb_evals2_vit_f128.py -m=work_dirs/softmax_ms1mv3_r18_per70class_nccl_128/model.pt -net=r18 -m_old=../work_dirs/iccv_2023/f128_softmax_ms1mv3_r18_per30_class/model.pt >> ./result_log/softmax_ms1mv3_r18_per70class_nccl_128_crosstest_11.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python ijb_evals2_vit_f128.py -m=work_dirs/softmax_ms1mv3_r18_per70class_nccl_128/model.pt -net=r18 -m_old=../work_dirs/iccv_2023/f128_softmax_ms1mv3_r18_per30_class/model.pt -N >> ./result_log/softmax_ms1mv3_r18_per70class_nccl_128_crosstest_1N.log 2>&1 &
