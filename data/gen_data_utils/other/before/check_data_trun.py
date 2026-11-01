import os
join = os.path.join
import tifffile as tiff
from PIL import Image
import numpy as np
import nibabel as nib
from tqdm import tqdm

def trans_tif_rgb(tif_data):
    # if(tif_data is None):
    #     print('=====================')
    fen_mu = (np.max(tif_data)-np.min(tif_data))
    # print(fen_mu)
    if(fen_mu==0):
        tif_data[:] = 0
    else:
        tif_data = (tif_data-np.min(tif_data))/fen_mu

    tif_data = tif_data * 255
    tif_data = tif_data.astype(np.uint8)
    tif_data = np.stack([tif_data, tif_data, tif_data], axis=-1)
    # print(tif_data.shape)
    return Image.fromarray(tif_data)

# if __name__ == "__main__":
#     src_path = '/data1/liuyang/stent/interior/data/costa/trun_data/radius-64/train_data'
#     save_path = '/data1/liuyang/stent/interior/data/costa/save_to_check/trun_data/radius-64/'
#     to_down  = ['25_4-fdkNoFilter']

#     for i in to_down:
#         vol_path = join(src_path,i)
#         # os.makedirs(save_dir,exist_ok=True)
#         for j in os.listdir(vol_path):
#             vol_train_gt_path = join(vol_path,j)
#             # print(join(vol_path,j))
#             save_dir = join(save_path,i,j)
#             for m in os.listdir(vol_train_gt_path):
#                 # proj = tiff.imread(join(vol_train_gt_path,m))
#                 proj = nib.load(join(vol_train_gt_path,m))
#                 proj_save_path = join(save_dir,m.split('.')[0])
#                 os.makedirs(proj_save_path,exist_ok=True)
#                 for k in index:
#                     k = int(k)
#                     trans_tif_rgb(proj[:,:,k]).save(join(proj_save_path,str(k).zfill(2)+'.jpg'))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="This script processes command-line arguments for a main function.")
    parser.add_argument("--radius", type=int, required=True)
    args = parser.parse_args()
    radius = args.radius
    src_path  = f'/data1/liuyang/stent/interior/data/costa/trun_data/radius-{radius}'
    save_path = f'/data1/liuyang/stent/interior/data/costa/save_to_check/trun_data/radius-{radius}'
    to_down  = ['45_4-fdkNoFilter']

    # skip_file = set(['IXI_Guys_074_sample_00'])

    
    for j in os.listdir(src_path): #j : train_data、test_data
        for i in to_down:
            vol_train_gt_path = join(src_path,j,i) # /data1/liuyang/stent/interior/data/costa/trun_data/radius-64/train_data/25_4-fdkNoFilter

            vol_train_path = join(vol_train_gt_path, 'vol_train')
            vol_gt_path = join(vol_train_gt_path, 'vol_gt')
            
            vol_train_list = sorted(os.listdir(vol_train_path))
            vol_gt_list = sorted(os.listdir(vol_gt_path))
            
            portion = 0.1
            to_be_processed_index = [int(i) for i in np.random.choice(len(vol_train_list), int(len(vol_train_list) * portion), replace=False)]
            vol_train_list = np.array(vol_train_list)[to_be_processed_index].tolist()
            vol_gt_list = np.array(vol_gt_list)[to_be_processed_index].tolist()

            save_dir = join(save_path,i,j,'vol_train') # save_dir = /data1/liuyang/stent/interior/data/costa/save_to_check/radius-64/train_data/proj_gt
            for m in tqdm(vol_train_list, desc='处理 vol_train'):  #vol_train_gt_path = /data1/liuyang/stent/interior/data/costa/src_data/radius-64/train_data
                # print(join(proj_path,m))
                # if(m.split('.')[0] in skip_file):
                #     print(f'skip: {m}')
                #     continue
                # proj = np.load(join(vol_train_path,m))['data']
                proj = nib.load(join(vol_train_path,m)).get_fdata()
                proj_save_path = join(save_dir,m.split('.')[0])
                os.makedirs(proj_save_path,exist_ok=True)
                # print(proj_save_path,proj.shape)
                for k in np.linspace(0, proj.shape[2], 20, endpoint=False):
                    k = int(k)
                    # print(proj[:,:,k].shape)
                    trans_tif_rgb(proj[:,:,k]).save(join(proj_save_path,str(k).zfill(2)+'.jpg'))

            save_dir = join(save_path,i,j,'vol_gt') # save_dir = /data1/liuyang/stent/interior/data/costa/save_to_check/radius-64/train_data/proj_gt
            for m in tqdm(vol_gt_list, desc='处理 vol_gt'):  #vol_train_gt_path = /data1/liuyang/stent/interior/data/costa/src_data/radius-64/train_data
                # if(m.split('.')[0] in skip_file):
                #     print(f'skip: {m}')
                #     continue
                proj = nib.load(join(vol_gt_path,m)).get_fdata()
                proj_save_path = join(save_dir,m.split('.')[0])
                os.makedirs(proj_save_path,exist_ok=True)
                for k in np.linspace(0, proj.shape[2], 20, endpoint=False):
                    k = int(k)
                    trans_tif_rgb(proj[:,:,k]).save(join(proj_save_path,str(k).zfill(2)+'.jpg'))
        print(i)