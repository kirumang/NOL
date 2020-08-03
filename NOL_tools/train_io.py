import numpy as np
def get_original_image(img):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(3):
        img[..., i]= img[..., i] *std[i]+ mean[i]
    return np.clip(img,0.,1.)

