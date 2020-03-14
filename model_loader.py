from model import Net
import torch


def load_model(upscale_factor = 2):
    if upscale_factor == 2:
        path = 'A:\Projects\CNNs\FSRCNN_train\FSRCNN\model_up_2_epoch100.pth'
    elif upscale_factor == 3:
        path = 'A:\Projects\CNNs\FSRCNN_train\FSRCNN\model_up_3_epoch150.pth'
    elif upscale_factor == 4:
        path = 'A:\Projects\CNNs\FSRCNN_train\FSRCNN\model_up_4_epoch150.pth'
    
    model = Net(num_channels = 1, upscale_factor=upscale_factor)
    model.load_state_dict(torch.load(path))
    
    return model