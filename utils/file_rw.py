import numpy as np
import os
import scipy.sparse as sp
import pickle

def read_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        config = pickle.load(f)
        return config

def read_sp(npz_path):
    return sp.load_npz(npz_path).toarray()

def write_sp(image, save_dir):

    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    sparse_img = sp.csr_matrix(image)
    sp.save_npz(save_dir, sparse_img, compressed=True)

def save_sparse_projections(
    proj: np.ndarray,
    out_dir: str,
    dtype=np.float32
):
    """
    proj: ndarray (Z, H, W)
    """
    os.makedirs(out_dir, exist_ok=True)
    Z, _, _ = proj.shape
    # meta = {
    #     "shape": [Z, H, W],
    #     "dtype": str(dtype)
    # }
    for z in range(Z):
        img = proj[z]

        # 转 CSR（核心）
        sparse_img = sp.csr_matrix(img.astype(dtype, copy=False))

        sp.save_npz(
            os.path.join(out_dir, f"proj_{z:03d}.npz"),
            sparse_img,
            compressed=True
        )

    # with open(os.path.join(out_dir, "meta.json"), "w") as f:
    #     json.dump(meta, f)
    # print(f"Saved {Z} sparse projections to {out_dir}")


def load_3d_from_z_indices(sparse_dir, indices, H=2048, W=2048):
    """
    sparse_dir: 存 proj_000.npz, proj_001.npz ... 的目录
    indices:    list[int]
    H, W:       图像尺寸
    """
    out = np.zeros((len(indices), H, W), dtype=np.float32)
    for i, z in enumerate(indices):
        sp_img = sp.load_npz(os.path.join(sparse_dir, f"proj_{z:03d}.npz"))
        out[i] = sp_img.toarray()
    return out


