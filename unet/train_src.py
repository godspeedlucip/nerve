import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
join = os.path.join
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from load_data import StentDataset
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# ====== 引入网络 & 损失 ======
from unet3d import UNet3D
# from unet3d_2 import UNet3D

from loss import CombinedLoss
import time

from utils.preprocess import save_model, resume_model, z_score_norminazation
# from utils.train_utils import PolyLRScheduler, DC_and_topk_loss

# ====== Dice指标 ======
def dice_score(preds, targets, threshold=0.5, eps=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum()
    return (2.0 * intersection + eps) / (preds.sum() + targets.sum() + eps)

# ====== 训练循环 ======
def train_model(num_epochs,work_name, proj_path,gt_path, save_dir='./output', batch_size=16, lr=1e-4, device="cuda",checkpoint_path=None):
    # 数据集划分：训练 / 验证 / 测试
    # proj_dir = '/data/liuyang/stent/DTR/data/120_120'

    print(f'vol_train_path: {proj_path}\n', 
        f'vol_gt_path: {gt_path}\n', 
        f'save_dir: {save_dir}\n',
        f'batch_size: {batch_size}\n', 
        f'device: {device}\n', 
        f'work_name: {work_name}',
        f'num_epochs: {num_epochs}')


    save_path = join(save_dir,work_name)
    model_save_path = join(save_path,'models')
    log_file_path = join(save_path,'training_log.log')
    os.makedirs(save_path,exist_ok=True)
    os.makedirs(model_save_path,exist_ok=True)
    writer = SummaryWriter(join(save_path,'tensorboard'))
    with open(log_file_path, 'a') as f:
        log_line = f"{num_epochs},{work_name},{proj_path},{gt_path},{batch_size},{checkpoint_path}\n"
        f.write(log_line)

    # dataset = StentDataset(proj_path,gt_path)
    patch_size=(128,128,128)
    # patch_size = (64,64,64)
    dataset = StentDataset(proj_path,gt_path,patch_size=patch_size) #新的数据用这个

    train_size = int(0.9 * len(dataset)) #选多一点的数据来训练
    val_size = len(dataset) - train_size
    test_size = 0
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # DataLoader
    print(num_epochs,work_name,proj_path,gt_path, save_dir,batch_size, lr, device, checkpoint_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 模型、损失、优化器
    model = UNet3D(in_channels=1, out_channels=1, base_channels=8).to(device)
    # model = UNet3D(num_channels=1).to(device)
    criterion = CombinedLoss(dice_weight=1.0, lap_weight=0.1).to(device) # loss也要移动到cuda上面
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))


    start_epoch = 0
    if(checkpoint_path is not None):
        model,optimizer,start_epoch = resume_model(checkpoint_path,model,optimizer,device)

    # 训练 & 验证
    print('start training...')
    pbar = tqdm(range(start_epoch,num_epochs),desc='Epoch: ')
    for epoch in pbar:
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}") #正确显示进度
        start_time = time.time()

        # --- 训练阶段 ---
        model.train()
        train_loss, train_dice = 0.0, 0.0
        # for inputs, targets in tqdm(train_loader,desc='load data in one epoch'):
        for inputs, targets in train_loader:
            # print(111)
            inputs, targets = inputs.to(device), targets.to(device)
            # print(inputs.shape)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice += dice_score(outputs, targets).item()
        

        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)

        save_model(epoch=epoch,model=model,optimizer=optimizer,model_save_path=model_save_path)
        # avg_loss =  avg_train_loss + avg_train_dice / 2
        # if(best_loss > avg_loss):
        #     best_loss = avg_loss
        #     torch.save(model, os.path.join(model_save_path,f"model_{str(epoch).zfill(4)}.pth"))

        # --- 验证阶段 ---
        model.eval()
        val_loss, val_dice = 0.0, 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                val_dice += dice_score(outputs, targets).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)

        # 第一个参数是标签，第二个是值，第三个是步数（这里用epoch）
        writer.add_scalar('val/loss', avg_val_loss, epoch)
        writer.add_scalar('val/dice', avg_val_dice, epoch)
        writer.add_scalar('train/loss', avg_train_loss, epoch)
        writer.add_scalar('train/dice', avg_train_dice, epoch)
        
        end_time = time.time()
        consume_times = end_time-start_time
        current_lr = optimizer.param_groups[0]['lr'] #从优化器中读取学习率并输出

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | Train Dice: {avg_train_dice:.4f} "
              f"| Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f}"
            #   f"当前的学习率： {current_lr: .2f}"
            #   f"训练时长: {consume_times/60} 分 {consume_times%60:.2f} 秒"
            f"  训练时长: {consume_times:.2f} 秒"
              )\
        
        # 写入日志文件
        with open(log_file_path, 'a') as f:
            log_line = f"{epoch+1},{avg_train_loss:.4f},{avg_train_dice:.4f},{avg_val_loss:.4f},{avg_val_dice:.4f},{consume_times:.2f}\n"
            f.write(log_line)

    # # --- 测试阶段 ---
    # print("\n测试集评估中...")
    # model.eval()
    # test_loss, test_dice = 0.0, 0.0
    # with torch.no_grad():
    #     for inputs, targets in test_loader:
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         outputs = model(inputs)
    #         loss = criterion(outputs, targets)

    #         test_loss += loss.item()
    #         test_dice += dice_score(outputs, targets).item()

    # avg_test_loss = test_loss / len(test_loader)
    # avg_test_dice = test_dice / len(test_loader)
    # print(f"Test Loss: {avg_test_loss:.4f} | Test Dice: {avg_test_dice:.4f}")

    writer.close() # 关闭tensorboard

    return model

# ====== 启动训练 ======
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
    parser.add_argument("--data_path",type=str)
    parser.add_argument("--work_name", type=str)
    parser.add_argument("--save_dir", type=str,default='./output')
    parser.add_argument("--batch_size", type=int,default=16)
    parser.add_argument("--checkpoint_path", type=str,default=None)
    parser.add_argument("--gt_path", type=str,default=None)
    parser.add_argument("--epochs", type=int,default=400)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('device: ',device)
    # work_name = '45_15'
    # proj_path = join('/data/liuyang/stent/DTR/data',work_name)

    # gt_dir = '/data1/liuyang/stent/data/15/vol'
    # gt_dir = '/data1/liuyang/stent/data/20/vol'
    gt_dir = args.gt_path
    proj_path = args.data_path
    work_name = args.work_name
    save_dir = args.save_dir
    batch_size = args.batch_size
    checkpoint_path = args.checkpoint_path
    epochs = args.epochs
    # print(proj_path,work_name,save_dir,checkpoint_path,epochs,batch_size)
    assert set(os.listdir(proj_path))==set(os.listdir(gt_dir))
    
    model = train_model(num_epochs=epochs, proj_path=proj_path,gt_path=gt_dir,work_name=work_name,save_dir=save_dir,batch_size=batch_size, lr=1e-4, device=device,checkpoint_path=checkpoint_path)


