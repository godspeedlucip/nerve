import numpy as np
import torch,os

def z_score_norminazation(image): #
    mean = image.mean()
    std = image.std()
    image -= mean
    image /= (max(std, 1e-8))
    return image

def rescale_0_1_norminazation(image): #将图像归一化到0-1的范围内
    image = np.copy(image)
    image -= np.min(image)
    image /= np.clip(np.max(image), a_min=1e-8, a_max=None)
    return image


def save_model(epoch,model,optimizer,model_save_path):
    # 如果想保存整个训练状态，包括优化器，可以这么做
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, os.path.join(model_save_path, f"checkpoint_{str(epoch).zfill(4)}.pth"))

def resume_model(checkpoint_path,model,optimizer,device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1 # 从下一轮开始
    print(f"Resuming training from epoch {start_epoch}")
    return model,optimizer,start_epoch
