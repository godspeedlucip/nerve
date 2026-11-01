CUDA_VISIBLE_DEVICES=4 python ../other/get_resample_mask.py   --target_spacing=0.2 --root_data_path=/data1/liuyang/stent/interior/data/other/topbrain_ct/root_data --helper_path=/data1/liuyang/stent/interior/data/other/topbrain_ct/helper &&
CUDA_VISIBLE_DEVICES=4 python ../other/get_distance_map.py   --target_spacing=0.2 --helper_path=/data1/liuyang/stent/interior/data/other/topbrain_ct/helper &&
CUDA_VISIBLE_DEVICES=4 python ../other/get_skeleton.py   --target_spacing=0.2  --helper_path=/data1/liuyang/stent/interior/data/other/topbrain_ct/helper



