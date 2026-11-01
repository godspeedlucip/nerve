import os
join = os.path.join
import tifffile as tiff
from PIL import Image
import numpy as np
import nibabel as nib

def trans_tif_rgb(tif_data):
    fen_mu = (np.max(tif_data)-np.min(tif_data))
    if(fen_mu==0):
        # tif_data = np.zeros_like()
        tif_data.fill(0)
        tif_data = tif_data.astype(np.uint8)
    else:
        tif_data = (tif_data-np.min(tif_data))/fen_mu
        tif_data = tif_data * 255
        tif_data = tif_data.astype(np.uint8)
        tif_data = np.stack([tif_data, tif_data, tif_data], axis=-1)
    return Image.fromarray(tif_data)

if __name__ == "__main__":
    src_path = '/data1/liuyang/stent/interior/data/costa/src_data'
    save_path = '//data1/liuyang/stent/interior/data/costa/save_to_check'
    to_down  = ['radius-64']

    skip_file = set(['IXI_Guys_074_sample_00'])

    index = np.linspace(0,111,20,endpoint=False)
    for i in to_down:
        vol_path = join(src_path,i)
        for j in os.listdir(vol_path): # vol_path = /data1/liuyang/stent/interior/data/costa/src_data/radius-64
            vol_train_gt_path = join(vol_path,j)
            save_dir = join(save_path,i,j)

            proj_path = join(vol_train_gt_path, 'proj_gt')
            vol_gt_path = join(vol_train_gt_path, 'vol_gt')

            # save_dir = join(save_path,i,j,'proj_gt') # save_dir = /data1/liuyang/stent/interior/data/costa/save_to_check/radius-64/train_data/proj_gt
            # for m in os.listdir(proj_path):  #vol_train_gt_path = /data1/liuyang/stent/interior/data/costa/src_data/radius-64/train_data
            #     # print(join(proj_path,m))
            #     if(m.split('.')[0] in skip_file):
            #         print(f'skip: {m}')
            #         continue
            #     proj = np.load(join(proj_path,m))['data']
            #     proj_save_path = join(save_dir,m.split('.')[0])
            #     os.makedirs(proj_save_path,exist_ok=True)
            #     for k in np.linspace(0, proj.shape[0], 20, endpoint=False):
            #         k = int(k)
            #         trans_tif_rgb(proj[k]).save(join(proj_save_path,str(k).zfill(2)+'.jpg'))

            save_dir = join(save_path,i,j,'vol_gt') # save_dir = /data1/liuyang/stent/interior/data/costa/save_to_check/radius-64/train_data/proj_gt
            for m in os.listdir(vol_gt_path):  #vol_train_gt_path = /data1/liuyang/stent/interior/data/costa/src_data/radius-64/train_data
                if(m.split('.')[0] in skip_file):
                    print(f'skip: {m}')
                    continue
                proj = nib.load(join(vol_gt_path,m)).get_fdata()
                proj_save_path = join(save_dir,m.split('.')[0])
                os.makedirs(proj_save_path,exist_ok=True)
                for k in np.linspace(0, proj.shape[2], 20, endpoint=False):
                    k = int(k)
                    trans_tif_rgb(proj[:,:,k]).save(join(proj_save_path,str(k).zfill(2)+'.jpg'))
        print(j)