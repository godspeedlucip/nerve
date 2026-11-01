import torch
from torch.optim.lr_scheduler import _LRScheduler
import torch.optim as optim

# =================================================================
# 1. PolyLRScheduler 类定义 (与您提供的代码相同)
# =================================================================

class PolyLRScheduler(_LRScheduler):
    """
    实现多项式学习率衰减 (Polynomial Learning Rate Decay)
    """
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0 # 内部步数计数器
        
        # 将优化器中所有参数组的初始学习率设置为 initial_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_lr
            
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step=None):
        """
        更新学习率。
        current_step 是指总的训练 Batch 数。
        """
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        # 多项式衰减公式：LR_new = LR_initial * (1 - current_step / max_steps) ^ exponent
        # 确保 current_step 不超过 max_steps，否则 new_lr 可能为负
        current_step = min(current_step, self.max_steps)
        
        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        
        # 将新的学习率应用到优化器的所有参数组
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        # 记录最新的学习率
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def get_last_lr(self):
        return self._last_lr

# =================================================================
# 2. 模拟训练配置和调度器使用
# =================================================================

def lr_scheduler_demo(initial_lr: float, total_epochs: int, batches_per_epoch: int, exponent: float):
    """
    演示不同 Epoch 下的学习率变化
    """
    print(f"--- 学习率调度演示 ---")
    print(f"初始学习率 (Initial LR): {initial_lr}")
    print(f"总 Epoch 数 (Total Epochs): {total_epochs}")
    print(f"每 Epoch Batch 数 (Batches/Epoch): {batches_per_epoch}")
    print(f"多项式指数 (Exponent): {exponent}")
    
    # 计算总的训练步数 (即 total batches)
    max_steps = total_epochs
    print(f"总训练步数 (Max Steps): {max_steps}\n")

    # 1. 模拟网络参数和优化器 (无需实际训练)
    # 假设有一个虚拟模型，只用于初始化优化器
    dummy_param = torch.nn.Parameter(torch.randn(1))
    optimizer = optim.Adam([dummy_param], lr=initial_lr)

    # 2. 初始化调度器
    scheduler = PolyLRScheduler(
        optimizer,
        initial_lr=initial_lr,
        max_steps=max_steps,
        exponent=exponent
    )

    # 3. 模拟训练循环
    current_step = 0
    
    # 打印表头
    print(f"{'Epoch':<5} | {'总步数':<8} | {'学习率 (LR)':<15}")
    print("-" * 30)

    for epoch in range(total_epochs):
        # 模拟一个 Epoch 中的 Batch 循环
        # for batch in range(batches_per_epoch):
            # 模拟梯度更新和优化器步骤
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            
            # 🌟 每次 Batch/Step 结束后更新学习率
        scheduler.step()
        current_step += 1

        # 打印当前 Epoch 结束时的学习率
        current_lr = scheduler.get_last_lr()[0]
        
        # 仅打印 Epoch 0, 50, 100, 200, ... 或最后一个 Epoch 的结果
        if epoch % 5 == 0 or epoch == total_epochs - 1:
            print(f"{epoch + 1:<5} | {current_step:<8} | {current_lr:.10f}")

# =================================================================
# 3. 运行演示
# =================================================================

if __name__ == '__main__':
    # 典型 nnU-Net 风格参数
    initial_lr = 0.01
    total_epochs = 400
    # 假设您的训练数据加载器有 200 个 Batch
    batches_per_epoch = 200 
    exponent = 0.9 

    lr_scheduler_demo(
        initial_lr=initial_lr,
        total_epochs=total_epochs,
        batches_per_epoch=batches_per_epoch,
        exponent=exponent
    )