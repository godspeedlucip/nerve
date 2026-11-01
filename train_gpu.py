import time
from datetime import datetime
import torch
import argparse


def occupy_gpu_if_free(gpu_id, target_frac=0.8, check_interval=20):
    """
    实时监控 GPU，如果可用显存 >= target_frac，则占用 target_frac 的总显存
    """
    torch.cuda.set_device(gpu_id)

    total_mem = torch.cuda.get_device_properties(gpu_id).total_memory
    target_mem = int(total_mem * target_frac)

    print(f"[GPU {gpu_id}] 总显存: {total_mem / 1024**2:.2f} MB")
    print(f"[GPU {gpu_id}] 目标占用: {target_mem / 1024**2:.2f} MB")

    dummy_tensor = None

    while True:
        free_mem, total_mem_now = torch.cuda.mem_get_info(gpu_id)
        free_ratio = free_mem / total_mem_now

        print(
            f"[{datetime.now()}] GPU {gpu_id} | "
            f"可用显存: {free_mem / 1024**2:.2f} MB "
            f"({free_ratio*100:.2f}%)"
        )

        # 满足条件且还没占用
        if free_ratio >= target_frac and dummy_tensor is None:
            print(f"[GPU {gpu_id}] 可用显存 ≥ {target_frac*100:.0f}%，开始占用显存")

            n_elements = target_mem // 4  # float32 = 4 bytes
            dummy_tensor = torch.empty(
                (n_elements,),
                device=f"cuda:{gpu_id}",
                dtype=torch.float32
            )

            print(f"[GPU {gpu_id}] 显存占用完成")
            break
        time.sleep(check_interval)

    return dummy_tensor


def burn_gpu_compute(gpu_id, matrix_size=4096*5, sleep_ms=3000):
    """
    持续进行矩阵乘法，占用 GPU 计算资源
    """
    device = f"cuda:{gpu_id}"

    print(f"[GPU {gpu_id}] 初始化计算张量 {matrix_size} x {matrix_size}")

    A = torch.randn((matrix_size, matrix_size), device=device)
    B = torch.randn((matrix_size, matrix_size), device=device)

    torch.cuda.synchronize()

    print(f"[GPU {gpu_id}] 开始持续计算，占用算力")

    while True:
        C = torch.matmul(A, B)
        C = torch.relu(C)

        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--frac", type=float, default=0.4)
    parser.add_argument("--interval", type=int, default=10)
    args = parser.parse_args()

    print(f"启动时间: {datetime.now()}")
    print(f"监控 GPU: {args.gpu}")
    print(f"目标占用比例: {args.frac}")

    gpu = args.gpu
    frac = args.frac

    dummy_tensor = occupy_gpu_if_free(gpu, frac)
    burn_gpu_compute(gpu)
    # 保持主线程存活
    while True:
        time.sleep(3600)
        print('存活一小时~')


if __name__ == "__main__":
    main()
