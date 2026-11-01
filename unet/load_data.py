import torch
from torch.utils.data import DataLoader, Dataset
import os
join = os.path.join
import tifffile as tiff
import numpy as np
import random

from scipy.ndimage import uniform_filter
def extract_max_cover_patch(mask, patch_size=(128, 64, 64)):
    """
    在 3D mask 数据中裁剪一个 patch，最大限度覆盖有效值 (1)。
    
    参数:
        mask: numpy.ndarray, shape=(D, W, H)，0/1 mask
        patch_size: tuple, 需要裁剪的 patch 尺寸 (pd, pw, ph)

    返回:
        patch: numpy.ndarray, 裁剪得到的 patch
        start: tuple, 裁剪区域的起始索引 (d, w, h)
    """
    D, W, H = mask.shape
    pd, pw, ph = patch_size
    
    if pd > D or pw > W or ph > H:
        raise ValueError("Patch 大小超过了输入体素的大小")

    # 用 uniform_filter 高效计算任意窗口内 1 的数量
    # 注意：uniform_filter 给的是平均值，所以要乘以窗口体积得到和
    patch_volume = pd * pw * ph
    summed = uniform_filter(mask.astype(np.float32), size=patch_size, mode="constant") * patch_volume

    # 找到最大值的位置
    max_pos = np.unravel_index(np.argmax(summed), summed.shape)

    # uniform_filter 输出的值是“居中”窗口的平均数，因此需要偏移一半
    d0 = max(0, min(D - pd, max_pos[0] - pd // 2))
    w0 = max(0, min(W - pw, max_pos[1] - pw // 2))
    h0 = max(0, min(H - ph, max_pos[2] - ph // 2))

    patch = mask[d0:d0+pd, w0:w0+pw, h0:h0+ph]
    return patch, (d0, w0, h0)

class ArrayPadder: # 假设这个方法在一个类中
    def pad_to_target_size_diffDimen(
        self,
        input_array, 
        target_size=(128, 128, 128),
        padding_value=0
    ) -> np.ndarray:
        """
        将三维 NumPy 数组填充 (Pad) 到目标尺寸。
        
        策略：
        1. 只填充输入尺寸小于目标尺寸的维度，尽可能居中。
        2. 对于输入尺寸大于或等于目标尺寸的维度，不进行填充，也不抛出异常。
           这意味着最终数组的某些维度可能会大于 target_size。

        Args:
            input_array: 输入的三维数组 (D, H, W)。
            target_size: 目标形状 (D_target, H_target, W_target)。
            padding_value: 用于填充的值 (默认为 0)。

        Returns:
            padded_array: 填充后的数组。其形状取决于输入数组和 target_size。
        """
        # 维度检查保持不变
        if input_array.ndim != 3:
            raise ValueError(f"输入数组必须是三维的，但它的维度是 {input_array.ndim}。")
        if len(target_size) != 3:
             raise ValueError(f"目标尺寸必须是三维的 (D, H, W)，但传入的是 {target_size}。")

        # 1. 初始化填充宽度列表
        pad_width = []
        
        # 2. 计算每个维度的填充量
        for input_dim, target_dim in zip(input_array.shape, target_size):
            
            if input_dim < target_dim:
                # 只有当输入小于目标时，才计算填充量
                
                # 需要填充的总量
                total_padding = target_dim - input_dim
                
                # 计算前部填充量 (尽量居中)
                pad_before = total_padding // 2
                
                # 计算后部填充量 (确保总和正确)
                pad_after = total_padding - pad_before
                
                pad_width.append((pad_before, pad_after))
                
            else: 
                # input_dim >= target_dim
                # 保持该维度不变，填充宽度为零
                pad_width.append((0, 0))

        # 3. 执行 np.pad 填充
        padded_array = np.pad(
            input_array, 
            pad_width=pad_width, 
            mode='constant', 
            constant_values=padding_value
        )
        
        return padded_array
    
    def pad_to_target_size_AllDimen(
        self,
        input_array, 
        target_size = (128, 128, 128),
        padding_value = 0
    ) -> np.ndarray:
        """
        将三维 NumPy 数组填充 (Pad) 到目标尺寸，使用零填充。
        填充策略：在每个轴上尽可能居中。

        Args:
            input_array: 输入的三维数组 (D, H, W)。
            target_size: 目标形状 (D_target, H_target, W_target)。
            padding_value: 用于填充的值 (默认为 0)。

        Returns:
            padded_array: 填充后的数组，形状为 target_size。
        """
        if input_array.ndim != 3:
            raise ValueError(f"输入数组必须是三维的，但它的维度是 {input_array.ndim}。")

        # 1. 初始化填充宽度列表
        pad_width = []
        
        # 2. 计算每个维度的填充量
        for input_dim, target_dim in zip(input_array.shape, target_size):
            if input_dim > target_dim:
                raise ValueError(f"输入尺寸 {input_dim} 大于目标尺寸 {target_dim}，无法填充。")

            # 需要填充的总量
            total_padding = target_dim - input_dim
            
            # 计算前部填充量 (尽量居中)
            pad_before = total_padding // 2
            
            # 计算后部填充量 (确保总和正确)
            pad_after = total_padding - pad_before
            
            pad_width.append((pad_before, pad_after))

        # 3. 执行 np.pad 填充
        # mode='constant' 表示使用一个常数值填充
        padded_array = np.pad(
            input_array, 
            pad_width=pad_width, 
            mode='constant', 
            constant_values=padding_value
        )
        
        return padded_array

class StentDataset(Dataset):
    def __init__(self, train_dir, gt_dir,patch_size =(128,128,128)):
        super().__init__()
        self.train_dir = train_dir
        self.gt_dir = gt_dir
        self.all_data = sorted(os.listdir(self.train_dir))
        self.gt_data = sorted(os.listdir(self.gt_dir))
        self.patch_size = patch_size # 新增的裁剪尺寸参数
        self.padder = ArrayPadder()


    def __len__(self):
        return len(self.all_data)

    def z_score_norminazation(self,image): #
        mean = image.mean()
        std = image.std()
        image -= mean
        image /= (max(std, 1e-8))
        return image

    def __getitem__(self, idx):
        train_data = tiff.imread(join(self.train_dir,self.all_data[idx]))
        gt_data = tiff.imread(join(self.gt_dir,self.gt_data[idx]))
        # print(self.all_data[idx],self.gt_data[idx])
        # assert(self.all_data[idx]==self.gt_data[idx])
        # random_integer = random.randint(0, 100)
        # return np.zeros([3,3,3]),np.zeros([3,3,3])

        pad_train_data = self.padder.pad_to_target_size_diffDimen(input_array=train_data, target_size=self.patch_size, padding_value=np.min(train_data))
        pad_gt_data = self.padder.pad_to_target_size_diffDimen(input_array=gt_data, target_size=self.patch_size, padding_value=np.min(gt_data))

        d, h, w = pad_train_data.shape
        p_d, p_h, p_w = self.patch_size

        start_d = random.randint(0, d - p_d)
        start_h = random.randint(0, h - p_h)
        start_w = random.randint(0, w - p_w)
        cropped_train = pad_train_data[start_d:start_d + p_d, start_h:start_h + p_h, start_w:start_w + p_w]
        cropped_gt = pad_gt_data[start_d:start_d + p_d, start_h:start_h + p_h, start_w:start_w + p_w]

        # print(cropped_train.shape, cropped_gt.shape)

        ## 将数据pad到 (128,128,128)
        # print(type(np.min(cropped_gt)))
        # cropped_gt = self.padder.pad_to_target_size_AllDimen(cropped_gt, (128,128,128), padding_value=np.min(cropped_gt))
        # cropped_train = self.padder.pad_to_target_size_AllDimen(cropped_gt,(128,128,128),padding_value=np.min(cropped_train))

        cropped_train = self.z_score_norminazation(cropped_train) #对图像进行z-score归一化

        cropped_train = np.expand_dims(cropped_train,axis=0)
        cropped_gt = np.expand_dims(cropped_gt,axis=0)
        # print('load data: ',cropped_train.shape,cropped_gt.shape) # (1,128,128,128)
        return cropped_train, cropped_gt
    

## 新的数据对应的数据集
# class StentDataset_2(Dataset):
#     def __init__(self, train_dir, gt_dir,patch_size =(128,128,128)):
#         super().__init__()
#         self.train_dir = train_dir
#         self.gt_dir = gt_dir
#         self.all_data = sorted(os.listdir(self.train_dir))
#         self.gt_data = sorted(os.listdir(self.gt_dir))
#         self.patch_size = patch_size # 新增的裁剪尺寸参数

#     def __len__(self):
#         return len(self.all_data)

#     def __getitem__(self, idx):
#         # train_data = np.load(join(self.train_dir,self.all_data[idx]))['data']
#         # gt_data = np.load(join(self.gt_dir,self.gt_data[idx]))['data']    
#         # train_data = tiff.imread(join(self.train_dir,self.all_data[idx]))
#         train_name = self.all_data[idx]
#         for i in self.gt_data:
#             if(train_name.rfind(i.split('.')[0])!=-1):
#                 # print(train_name,i)
#                 gt_data = tiff.imread(join(self.gt_dir,i))
        
#         # return np.zeros([3,3]),np.zeros([3,3])

#         d, h, w = train_data.shape
#         p_d, p_h, p_w = self.patch_size
#         start_d = random.randint(0, d - p_d)
#         start_h = random.randint(0, h - p_h)
#         start_w = random.randint(0, w - p_w)

#         # 4. 根据起始坐标和patch_size进行裁剪
#         cropped_train = train_data[start_d:start_d + p_d, start_h:start_h + p_h, start_w:start_w + p_w]
#         cropped_gt = gt_data[start_d:start_d + p_d, start_h:start_h + p_h, start_w:start_w + p_w]
#         # print(1)
        
#         cropped_train = np.expand_dims(cropped_train,axis=0)
#         cropped_gt = np.expand_dims(cropped_gt,axis=0)
#         # print('load data: ',cropped_train.shape,cropped_gt.shape) # (1,128,128,128)
#         return cropped_train, cropped_gt


class StentDataset_test(Dataset):
    def __init__(self, train_dir,gt_dir,patch_size =(256,128,128)):
        super().__init__()
        self.train_dir = train_dir
        self.gt_dir = gt_dir
        self.all_data = sorted(os.listdir(self.train_dir))
        self.patch_size = patch_size # 新增的裁剪尺寸参数

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        # train_data = tiff.imread(join(self.train_dir,self.all_data[idx]))
        # gt_data = tiff.imread(join(self.gt_dir,self.gt_data[idx]))
        
        # npz文件存储
        # train_data = np.load(join(self.train_dir,self.all_data[idx]))['data']
        train_data = tiff.imread(join(self.train_dir,self.all_data[idx]))
        gt_data = tiff.imread(join(self.gt_dir,self.all_data[idx]))

        d, h, w = train_data.shape
        p_d, p_h, p_w = self.patch_size 
        start_d = random.randint(0, d - p_d)
        start_h = random.randint(0, h - p_h)
        start_w = random.randint(0, w - p_w)

        # 4. 根据起始坐标和patch_size进行裁剪
        cropped_train = train_data[start_d:start_d + p_d, start_h:start_h + p_h, start_w:start_w + p_w]
        cropped_gt = gt_data[start_d:start_d + p_d, start_h:start_h + p_h, start_w:start_w + p_w]
        cropped_train = np.expand_dims(cropped_train,axis=0)
        cropped_gt = np.expand_dims(cropped_gt,axis=0)

        print('load data: ',cropped_train.shape,cropped_gt.shape)
        return cropped_train,cropped_gt
    

if __name__ == "__main__":
    proj_path = '/data1/liuyang/stent/DTR/data/train_data/new/90_4-20-fdkNoFilter/vol_train'
    gt_path = '/data1/liuyang/stent/DTR/data/train_data/new/90_4-20-fdkNoFilter/vol_gt'
    dataset = StentDataset(proj_path,gt_path)
    train_size = int(0.9 * len(dataset)) #选多一点的数据来训练
    val_size = len(dataset) - train_size
    test_size = 0
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    for inputs, targets in train_loader:
        pass
        
