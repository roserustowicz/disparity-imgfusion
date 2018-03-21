'''
Fuses two images with a disparity map
Author: Rose Rustowicz
Date: 16 March 2018
'''

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import rgb_to_hsv as rgb2hsv
from matplotlib.colors import hsv_to_rgb as hsv2rgb

def shift_img(img, disp_map):
    shifted_img = np.zeros((img.shape))

    for r in range(disp_map.shape[0]):
        for c in range(disp_map.shape[1]):
            shift = np.round(disp_map[r, c])
            if c-shift < 0:
                pass
            else:
                shifted_img[r,c-shift,:] = img[r, c, :]
    return shifted_img

# Read in color corrected images
img1_cc = np.asarray(Image.open("../rectified_00_rgb/07rectified_rgbcc.png")) / 255.
img2_cc = np.asarray(Image.open("../rectified_01_pan/07rectified_pancc.png")) / 255.

# Use the mean across all channels for the RGB image, and any channel of the 
#  panchromatic image (all channels of the panchromatic image are the same)
img2_cc = img2_cc[:,:,1]
img1_cc = cv2.resize(img1_cc, (960, 720))
img2_cc = cv2.resize(img2_cc, (960, 720))

# Load disparity map
disp_map = np.load('results_dispmaps/disparity_SGBM_blocksize_filt_23.npy').astype(int)
disp_map[disp_map < 0] = 0

plt.figure()
plt.imshow(img1_cc)
plt.show()

plt.figure()
plt.imshow(img2_cc, cmap='gray')
plt.show()

# Shift image by disparity map and show
shifted_img = shift_img(img1_cc, disp_map)
plt.figure()
plt.imshow(shifted_img)
plt.show()

# If hole, replace with average of local neighborhood
holes = np.array(np.where(shifted_img == 0)).T
for index in range(holes.shape[0]):
    inds = holes[index,:]
    shifted_img[inds[0], inds[1], inds[2]] = np.mean(img1_cc[inds[0]-5:inds[0]+5, inds[1]-5:inds[1]+5, :])

plt.figure()
plt.imshow(shifted_img)
plt.show()

# Convert RGB image to HSV, average the V channel with the shifted
#  panchromatic image, and convert back to RGB to yield the fused image
hsv_img = rgb2hsv(shifted_img)
hsv_img[:,:,2] = (hsv_img[:,:,2] + img2_cc)/2.
fused_img = hsv2rgb(hsv_img)

plt.figure()
plt.imshow(fused_img)
plt.show()

fused_img = np.nan_to_num(fused_img)
fused_img = (fused_img - np.min(fused_img)) / (np.max(fused_img) - np.min(fused_img))
fused_img = (fused_img*255).astype(np.uint8)

fused_img = cv2.cvtColor(fused_img, cv2.COLOR_BGR2RGB)
cv2.imwrite("fused_img.png", fused_img)
