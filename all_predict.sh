### 让nnunet去预测所有ICBM_010/ICBM_010_sample_000这样的文件结构的数据

# 输入数据的根目录 (包含 image_proj 的文件夹)
SOURCE_DIR="/data1/liuyang/stent/DTE/data/resample_data/costa/voxel_0.2/all_proj/radius-64/vessel/train_data/image_proj"
TARGET_DIR="/data1/liuyang/stent/DTE/data/resample_data/costa/voxel_0.2/all_proj/radius-64/vessel/train_data/predict"
mkdir -p "$TARGET_DIR"
# 遍历源目录下的所有子文件夹
for subfolder in "$SOURCE_DIR"/*/; do
    # 去掉路径末尾的斜杠，获取子文件夹名称（例如 ICBM_010）
    folder_name=$(basename "$subfolder")
    output_path="$TARGET_DIR/$folder_name"
    mkdir -p "$output_path"
    echo $subfolder
    echo "正在处理: $folder_name ..."
    CUDA_VISIBLE_DEVICES=5 nnUNetv2_predict -i $subfolder -o  $output_path -d 801 -c 2d -f 1
done

SOURCE_DIR="/data1/liuyang/stent/DTE/data/resample_data/costa/voxel_0.2/all_proj/radius-64/vessel/test_data/image_proj"
TARGET_DIR="/data1/liuyang/stent/DTE/data/resample_data/costa/voxel_0.2/all_proj/radius-64/vessel/test_data/predict"
mkdir -p "$TARGET_DIR"
# 遍历源目录下的所有子文件夹
for subfolder in "$SOURCE_DIR"/*/; do
    # 去掉路径末尾的斜杠，获取子文件夹名称（例如 ICBM_010）
    folder_name=$(basename "$subfolder")
    output_path="$TARGET_DIR/$folder_name"
    mkdir -p "$output_path"
    
    echo "正在处理: $folder_name ..."
    CUDA_VISIBLE_DEVICES=5 nnUNetv2_predict -i $subfolder -o  $output_path -d 801 -c 2d -f 1

done
echo "所有预测任务已完成！结果保存在: $TARGET_DIR"
