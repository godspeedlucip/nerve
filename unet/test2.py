import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
join = os.path.join
import argparse
from natsort import natsorted

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from load_data import StentDataset,StentDataset_test,ArrayPadder
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# ====== 引入网络 & 损失 ======
from unet3d import UNet3D
import tifffile as tiff
import tigre,random
from loss import CombinedLoss
from utils.preprocess import z_score_norminazation


# def resume_model(checkpoint_path,model,optimizer,device):
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     start_epoch = checkpoint['epoch'] + 1 # 从下一轮开始
#     print(f"Resuming training from epoch {start_epoch}")
#     return model,optimizer,start_epoch

def dice_score(preds, targets,  eps=1e-6):
    intersection = (preds * targets).sum()
    return (2.0 * intersection + eps) / (preds.sum() + targets.sum() + eps)

def set_geo():
    # Lets create a geometry object
    geo = tigre.geometry()
    # VARIABLE                                   DESCRIPTION                    UNITS
    # -------------------------------------------------------------------------------------
    # Distances
    geo.DSD = 1250  # Distance Source Detector      (mm)
    geo.DSO = 750  # Distance Source Origin        (mm)
    # Detector parameters
    geo.nDetector = np.array([512,512])  # number of pixels              (px)
    geo.dDetector = np.array([0.2, 0.2])  # size of each pixel            (mm)
    geo.sDetector = geo.nDetector * geo.dDetector  # total size of the detector    (mm)
    # Image parameters
    geo.nVoxel = np.array([256, 128, 128])  # number of voxels              (vx)
    geo.dVoxel = np.array([0.2, 0.2, 0.2])  # size of each voxel            (mm)
    geo.sVoxel = geo.dVoxel * geo.nVoxel  # total size of the image       (mm)
    
    # Offsets
    geo.offOrigin = np.array([0, 0, 0])  # Offset of image from origin   (mm)
    # geo.offOrigin = geo.sVoxel /2 

    geo.offDetector = np.array([0, 0])  # Offset of Detector            (mm)
    # geo.offDetector = geo.sDetector / 2

    geo.accuracy = 0.5  # Variable to define accuracy of
    geo.COR = 0  # y direction displacement for
    geo.rotDetector = np.array([0, 0, 0])  # Rotation of the detector, by
    # geo.mode = "cone"  # Or 'parallel'. Geometry type.
    geo.mode = "cone"  # Or 'parallel'. Geometry type.
    return geo


geo = set_geo()
angles = np.linspace(0,np.pi*2,360)

# ====== 测试函数 ======
def test_model(model_path, data_path, gt_path, work_name, save_path, patch_size = (128,128,128), thred=0.5, device="cuda", if_project=False):
    def proceed_data(data_path, target_patch=(128,128,128), device="cuda"):
        padder = ArrayPadder()
        data = tiff.imread(data_path)
        data_pad = padder.pad_to_target_size_diffDimen(data, target_size=target_patch, padding_value=np.min(data)) #pad之后害需要裁剪

        data_pad = z_score_norminazation(data_pad)

        d, h, w = data_pad.shape
        p_d, p_h, p_w = target_patch

        start_d = random.randint(0, d - p_d)
        start_h = random.randint(0, h - p_h)
        start_w = random.randint(0, w - p_w)
        cropped_data_pad = data_pad[start_d:start_d + p_d, start_h:start_h + p_h, start_w:start_w + p_w]

        input = np.expand_dims(cropped_data_pad,axis=0)
        input = np.expand_dims(input,axis=0) # (1,1,128,128,128)
        input = torch.tensor(input).float()
        input = input.to(device)
        return input
    """
    加载模型并对测试集进行评估。

    Args:
        model_path (str): 保存的模型路径。
        proj_path (str): 投影数据的根目录。
        gt_path (str): GT标签数据的根目录。
        batch_size (int): 测试批次大小。
        device (str): 运行设备。
    """

    print(f'model path: {model_path}\n', 
          f'vol_train_path: {data_path}\n', 
          f'vol_gt_path: {gt_path}\n', 
          f'save_path: {save_path}\n', 
          f'thred: {thred}\n', 
          f'device: {device}\n', 
          f'work_name: {work_name}')

    # 1. 加载模型
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        return
    
    model = UNet3D(in_channels=1, out_channels=1, base_channels=8).to(device)
    criterion = CombinedLoss(dice_weight=1.0, lap_weight=0.1).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    # print(f"正在从 {model_path} 加载模型...")

    model.eval()  # 切换到评估模式，这将关闭Dropout和BatchNorm等

    vol_save_path = join(save_path,work_name,'vol')
    proj_save_path = join(save_path,work_name,'proj')
    os.makedirs(vol_save_path,exist_ok=True)
    os.makedirs(proj_save_path,exist_ok=True)

    # 2. 准备测试数据集
    # dataset = StentDataset_test(data_path)
    # batch_size = 1
    # test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    geo = set_geo()
    angles = np.linspace(0,np.pi*2,360)

    total_loss = 0
    total_dice = 0
    nums = 0
    with torch.no_grad():  # 禁用梯度计算，加速推理并节省内存
        # for inputs in tqdm(test_loader, desc="测试中"):
        for _,i in enumerate(sorted(os.listdir(data_path))):
            ## 直接把bool的mask转换为int好像就会成为全0的结果
            
            ## 这里把train和gt同时进行了填充，所有他们的shape都是一样的
            train_data = proceed_data(join(data_path,i), target_patch=patch_size, device='cuda')
            gt_data = proceed_data(join(gt_path,i),  target_patch=patch_size, device='cuda')
            output = torch.sigmoid(model(train_data))

            ## 目前用0.9的阈值
            mask = (output > thred)  # GPU上得到0/1，在cpu上就会很慢
            total_loss = criterion(mask, gt_data)
            total_dice += dice_score(mask, gt_data).item()
            # print(torch.min(mask),torch.max(mask))
            
            output_cpu = mask[0].cpu().numpy().squeeze()
            tiff.imwrite(join(vol_save_path,i),output_cpu.astype(np.float32))
            nums += 1
            if(if_project):
                proj = tigre.Ax(output_cpu, geo, angles)
                tiff.imwrite(join(proj_save_path,i),proj)

    return total_loss/nums, total_dice/nums


def project(vol,geo,angles):
    projections = tigre.Ax(vol.astype(np.float32), geo, angles)
    return projections

# ====== 启动测试 ======
if __name__ == "__main__":

    # models_prev = []
    # models_prev.append('/data1/liuyang/stent/interior/output/train/25-4-no-bs-8/models')
    # models_prev.append('/data1/liuyang/stent/interior/output/train/25-4-no-bs-16/models')
    # models_prev.append('/data1/liuyang/stent/interior/output/train/25-4-no-bs-32/models')
    # models_prev.append('/data1/liuyang/stent/interior/output/train/30-4-no-bs-16/models')
    # models_prev.append('/data1/liuyang/stent/interior/output/train/45-4-no-bs-16/models')
    # models_prev.append('/data1/liuyang/stent/interior/output/train/60-4-no-bs16/models')
    # models_prev.append('/data1/liuyang/stent/interior/output/train/90-4-no-bs-16/models')

    # datas = []
    # datas.append('/data1/liuyang/stent/interior/data/trun_test_data/25_4-fdkNoFilter')
    # datas.append('/data1/liuyang/stent/interior/data/trun_test_data/25_4-fdkNoFilter')
    # datas.append('/data1/liuyang/stent/interior/data/trun_test_data/25_4-fdkNoFilter')
    # datas.append('/data1/liuyang/stent/interior/data/trun_test_data/30_4-fdkNoFilter')
    # datas.append('/data1/liuyang/stent/interior/data/trun_test_data/45_4-fdkNoFilter')
    # datas.append('/data1/liuyang/stent/interior/data/trun_test_data/60_4-fdkNoFilter')
    # datas.append('/data1/liuyang/stent/interior/data/trun_test_data/90_4-fdkNoFilter')
    # save_dir = '/data1/liuyang/stent/interior/output/test'
    # thred = 0.5



    ## 配置多端训练
    models = []
    # models.append('/data1/liuyang/stent/interior/output/train/radius-30/25-4-no-bs-16/models')
    # models.append('/data1/liuyang/stent/interior/output/train/radius-60/25-4-no-bs-16/models')
    # models.append('/data1/liuyang/stent/interior/output/train/radius-60/25-5-no-bs-16/models')
    # models.append('/data1/liuyang/stent/interior/output/train/radius-60/90-4-no-bs16/models')
    # models.append('/data1/liuyang/stent/interior/output/train/radius-60/45-4-no-bs16/models')
    models.append('/data1/liuyang/stent/interior/output/train/radius-64/25_4-no-bs16/models')
    # models.append('/data1/liuyang/stent/interior/output/train/radius-64/45_4-no-bs16/models')
    # models.append('')


    datas = []  
    # datas.append('/data1/liuyang/stent/interior/data/trun_data/radius-30/test_data/25_4-fdkNoFilter')
    # datas.append('/data1/liuyang/stent/interior/data/trun_data/radius-60/test_data/25_4-fdkNoFilter')
    # datas.append('/data1/liuyang/stent/interior/data/trun_data/radius-60/test_data/25_5-fdkNoFilter')
    # datas.append('/data1/liuyang/stent/interior/data/trun_data/radius-60/test_data/90_4-fdkNoFilter')
    # datas.append('/data1/liuyang/stent/interior/data/trun_data/radius-60/test_data/45_4-fdkNoFilter')
    datas.append('/data1/liuyang/stent/interior/data/trun_data/radius-64/test_data/25_4-fdkNoFilter')
    # datas.append('/data1/liuyang/stent/interior/data/trun_data/radius-64/test_data/45_4-fdkNoFilter')
    # datas.append('')

    save_dirs = []
    # save_dirs.append('/data1/liuyang/stent/interior/output/test/radius-30')
    # save_dirs.append('/data1/liuyang/stent/interior/output/test/radius-60')
    # save_dirs.append('/data1/liuyang/stent/interior/output/test/radius-60')
    # save_dirs.append('/data1/liuyang/stent/interior/output/test/radius-60')
    # save_dirs.append('/data1/liuyang/stent/interior/output/test/radius-60')
    save_dirs.append('/data1/liuyang/stent/interior/output/test/radius-64')
    # save_dirs.append('/data1/liuyang/stent/interior/output/test/radius-64')
    # save_dirs.append('')

    thred = 0.5
    patch_size = []
    # patch_size.append((64,64,64))
    # patch_size.append((128,128,128))
    # patch_size.append((128,128,128))
    # patch_size.append((128,128,128))
    # patch_size.append((128,128,128))
    patch_size.append((128,128,128))
    # patch_size.append((128,128,128))
    # patch_size.append()

    
    for i in range(len(models)):
        # if(i not in to_be_deal_prev):
        #     continue
        if(os.path.isfile(models[i])):
            model_path = models[i]
        else:
            # model_root = join(models_prev[i],'models')
            model_root = models[i]
            model_path = join(model_root,sorted(os.listdir(model_root))[-1])
        assert model_path.endswith('.pth')

        # data_name = datas[i]
        # data_path = join(data_root,data_name,'vol_train')
        # gt_path = join(data_root,data_name,'vol_gt')
        data_name = os.path.basename(datas[i])
        data_path = join(datas[i],'vol_train')
        gt_path = join(datas[i],'vol_gt')

        work_model_name = os.path.dirname(model_path).split('/')[-2]
        work_name =  f"{work_model_name}-for-{data_name}"
        log_file_path = join(save_dirs[i],work_name, 'test_res.log')
        # print(data_path,gt_path,model_path,work_name,save_dir)
        total_loss,total_dice = test_model(model_path, data_path, gt_path, work_name, save_dirs[i], thred=thred, patch_size=patch_size[i], device="cuda",if_project=False)

        with open(log_file_path, 'w') as f:
            para_line = f"data_path: {data_path}, model_path: {model_path}, work_name: {work_name}, save_dir: {save_dirs[i]}, gt_path: {gt_path}, thred: {thred}\n"
            log_line = f"{total_loss:.4f},{total_dice:.4f}\n"
            f.write(para_line)
            f.write(log_line)

        print('total_loss: ',total_loss, ' total_dice: ',total_dice)

