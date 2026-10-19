import numpy as np


class StepSchedule:
    def __init__(self, num_epochs, step_length):
        self.step_length = step_length
        self.num_steps = num_epochs // step_length
        self.num_epochs = num_epochs + step_length

    def __call__(self, epoch):
        if epoch >= self.num_epochs - self.step_length:
            return 0
        step = epoch // self.step_length
        return max(0, 1 - step / self.num_steps)


class CosineSchedule:
    def __init__(self, num_epochs, min_val=0, max_val=1):
        self.num_epochs = num_epochs - 1
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, epoch):
        cos_out = (1 + np.cos(np.pi * epoch / self.num_epochs)) / 2
        return self.min_val + (self.max_val - self.min_val) * cos_out


class LinearSchedule:
    def __init__(self, num_epochs, init_pause):
        if num_epochs <= init_pause:
            raise ValueError("The number of epochs must be greater than the initial pause.")
        self.num_epochs = num_epochs - 1
        self.init_pause = init_pause

    def __call__(self, epoch):
        if epoch > self.num_epochs:
            raise ValueError("The current epoch is greater than the total number of epochs.")
        if epoch > self.init_pause:
            return min(1, max(0, 1.0 - (float(epoch - self.init_pause) / (self.num_epochs - self.init_pause))))
        else:
            return 1.0


class AlphaSchedule:
    def __init__(self, n_epochs, schedule, **kwargs):
        self.schedule = schedule
        self.linear = LinearSchedule(n_epochs, init_pause=kwargs["init_pause"])
        self.step = StepSchedule(n_epochs - kwargs["step_length"], step_length=kwargs["step_length"])
        self.cosine = CosineSchedule(n_epochs)

    def __call__(self, epoch):
        if self.schedule == "linear":
            return self.linear(epoch)
        elif self.schedule == "step":
            return self.step(epoch)
        elif self.schedule == "cosine":
            return self.cosine(epoch).astype('float32')
        else:
            raise ValueError("Enter valid schedule type")
