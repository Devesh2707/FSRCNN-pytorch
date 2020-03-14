from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np
import torch
from model_loader import load_model

img_to_tensor = ToTensor()
FSRCNN = load_model(upscale_factor=4)
cuda = torch.cuda.is_available()
if cuda:
    FSRCNN = FSRCNN.cuda()
def convert_frame(frame,w=None,h=None,image=False):
    if image:
        img = Image.open(frame).convert('YCbCr')
        w,h = img.size
        y,cb,cr = img.split()
    else:
        y,cb,cr = Image.fromarray(frame).convert('YCbCr').split()
    input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])
    if cuda:
        input = input.cuda()
        
    out = FSRCNN(input)
    del input
    out = out.cpu()
    out_img_y = out[0].detach().numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    out_img = out_img.resize(size=(w,h), resample = Image.LANCZOS)
    
    if image:
        return out_img
    else:
        out_arr = np.array(out_img)
        return out_arr