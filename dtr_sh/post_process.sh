## 合并到完整的投影
python /data1/liuyang/stent/DTE/gen_data/get_dtr_proj_concat.py --data_name costa_new --data_type vessel &
python /data1/liuyang/stent/DTE/gen_data/get_dtr_proj_concat.py --data_name costa_new --data_type stent &&


## 进行重建
CUDA_VISIBLE_DEVICES=2  python /data1/liuyang/stent/DTE/gen_data/get_dtr_recon.py --data_name costa_new --data_type vessel --tar_range 180 --tar_num 180 --use_split 0 &
CUDA_VISIBLE_DEVICES=2  python /data1/liuyang/stent/DTE/gen_data/get_dtr_recon.py --data_name costa_new --data_type stent  --tar_range 180 --tar_num 180 --use_split 0

