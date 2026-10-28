# 设置环境变量和基础路径
export CUDA_VISIBLE_DEVICES=3
## 变量的等号两边不要有空格，不然会被认为是命令
data_name="costa_new"
root_dir="/data1/liuyang/stent/DTE/data/resample_data"

BASE_DIR="${root_dir}/${data_name}/voxel_0.2/dtr/recon"
ARGS="-c 3d_fullres -f 1"

# Vessel 预测 (Dataset 904)
for split in train_data test_data; do
    nnUNetv2_predict -i $BASE_DIR/vessel/radius-64/$split/60_3-fdkNoFilter/predict/src \
                     -o $BASE_DIR/vessel/radius-64/$split/60_3-fdkNoFilter/predict/pred \
                     -d 906 $ARGS &
done

wait
echo "All predictions finished."
## 之后要get_dtr_map将得到的二值图像和原始图像相乘得到输入dte_map的数据






