# python /data1/liuyang/stent/DTE/gen_data/get_dtr_map.py --data_name costa_new  --data_type=vessel &
# python /data1/liuyang/stent/DTE/gen_data/get_dtr_map.py --data_name costa_new  --data_type=stent &&


data_name="smile2023"
export CUDA_VISIBLE_DEVICES=3

root_dir="/data1/liuyang/stent/DTE/data/resample_data"
PYTHON_EXEC="/data1/liuyang/stent/DTE/model/test.py"
CKPT="/data1/liuyang/stent/DTE/model/output/stent/best.pt"
BASE_PROJ="${root_dir}/${data_name}/voxel_0.2/dtr/proj"

# 2. 使用嵌套循环处理所有组合
# for task in vessel stent; do
for task in stent; do
    for split in train_data test_data; do
        
        # 构造当前路径
        CUR_DIR="${BASE_PROJ}/${task}/radius-64/${split}"
        
        echo "Running inference for: ${task} - ${split}..."
        
        python ${PYTHON_EXEC} infer \
            --ckpt "/data1/liuyang/stent/DTE/model/output/${task}/best.pt" \
            --in_img "${CUR_DIR}/predict/src_label" \
            --in_lbl "${CUR_DIR}/label/src_label" \
            --out_dir "${CUR_DIR}/predict/depth_label"
    done
done

echo "All tasks completed."

# CUDA_VISIBLE_DEVICES=2 python /data1/liuyang/stent/DTE/model/test.py  infer \
#  --ckpt /data1/liuyang/stent/DTE/model/output/stent/best.pt \
#  --in_img /data1/liuyang/stent/DTE/data/resample_data/costa_new/voxel_0.2/full_dtr/proj/vessel/radius-64/train_data/predict/src_label \
#  --in_lbl /data1/liuyang/stent/DTE/data/resample_data/costa_new/voxel_0.2/full_dtr/proj/vessel/radius-64/train_data/label/src_label \
#  --out_dir /data1/liuyang/stent/DTE/data/resample_data/costa_new/voxel_0.2/full_dtr/proj/vessel/radius-64/train_data/predict/depth_label &&


# CUDA_VISIBLE_DEVICES=2 python /data1/liuyang/stent/DTE/model/test.py  infer \
#  --ckpt /data1/liuyang/stent/DTE/model/output/stent/best.pt \
#  --in_img /data1/liuyang/stent/DTE/data/resample_data/costa_new/voxel_0.2/full_dtr/proj/stent/radius-64/train_data/predict/src_label \
#  --in_lbl /data1/liuyang/stent/DTE/data/resample_data/costa_new/voxel_0.2/full_dtr/proj/stent/radius-64/train_data/label/src_label \
#  --out_dir /data1/liuyang/stent/DTE/data/resample_data/costa_new/voxel_0.2/full_dtr/proj/stent/radius-64/train_data/predict/depth_label &&



#  CUDA_VISIBLE_DEVICES=2 python /data1/liuyang/stent/DTE/model/test.py  infer \
#  --ckpt /data1/liuyang/stent/DTE/model/output/stent/best.pt \
#  --in_img /data1/liuyang/stent/DTE/data/resample_data/costa_new/voxel_0.2/full_dtr/proj/vessel/radius-64/test_data/predict/src_label \
#  --in_lbl /data1/liuyang/stent/DTE/data/resample_data/costa_new/voxel_0.2/full_dtr/proj/vessel/radius-64/test_data/label/src_label \
#  --out_dir /data1/liuyang/stent/DTE/data/resample_data/costa_new/voxel_0.2/full_dtr/proj/vessel/radius-64/test_data/predict/depth_label &&


# CUDA_VISIBLE_DEVICES=2 python /data1/liuyang/stent/DTE/model/test.py  infer \
#  --ckpt /data1/liuyang/stent/DTE/model/output/stent/best.pt \
#  --in_img /data1/liuyang/stent/DTE/data/resample_data/costa_new/voxel_0.2/full_dtr/proj/stent/radius-64/test_data/predict/src_label \
#  --in_lbl /data1/liuyang/stent/DTE/data/resample_data/costa_new/voxel_0.2/full_dtr/proj/stent/radius-64/test_data/label/src_label \
#  --out_dir /data1/liuyang/stent/DTE/data/resample_data/costa_new/voxel_0.2/full_dtr/proj/stent/radius-64/test_data/predict/depth_label

