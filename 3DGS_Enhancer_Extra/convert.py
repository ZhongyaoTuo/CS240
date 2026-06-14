import os
import cv2
    
import torch
import torchvision.io as io
from torchvision.utils import save_image


if __name__ == "__main__":
    
    
    path = "/workspace/results_train/run1_relpose/training_512_mv/images/val/016/val/gs3583_idx7_room_0.5-renders.jpg"
    
    img = cv2.imread(path)
    
    new_path = path.replace(".jpg", "_fixed.jpg")
    # cv2.imwrite(new_path, img)

    # 1) Load the wrongly scaled image
    #    io.read_image(...) loads a uint8 tensor in [0,255],
    #    so we divide by 255.0 => floats in [0,1].
    wrong = io.read_image(path).float() / 255.0  # shape: [C,H,W], values in [0.5,1.0]

    # 2) Map [0.5,1] back down to [0,1].
    fixed_0_1 = 2.0 * (wrong - 0.5)  # Now the darkest pixel is 0, brightest is 1.

    # (Optionally) if you want it in [-1,1], do:
    # fixed_neg1_pos1 = 4.0 * wrong - 3.0

    # 3) Save the fixed image in normal [0,1] range as a PNG/JPG
    save_image(fixed_0_1, new_path)