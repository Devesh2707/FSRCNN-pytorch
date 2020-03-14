from model import Net
import torch


def load_model(upscale_factor = 2):
    if upscale_factor == 2:
        path = 'checkpoints\model_up_{}_epoch_150.pth'.format(upscale_factor)
    elif upscale_factor == 3:
        path = 'checkpoints\model_up_{}_epoch_150.pth'.format(upscale_factor)
    elif upscale_factor == 4:
        path = 'checkpoints\model_up_{}_epoch_150.pth'.format(upscale_factor)
    
    model = Net(num_channels = 1, upscale_factor=upscale_factor)
    model.load_state_dict(torch.load(path))
    
    return model