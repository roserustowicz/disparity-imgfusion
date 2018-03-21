'''
Implementation of synthetic refocusing, given an input fused image 
and disparity map. Each pixel is blurred as a function of the disparity 
value in the corresponding pixel location of the disparity map.

Author: Rose Rustowicz
Date: 16 March 2018
'''

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.ndimage.filters import gaussian_filter as gauss_filt

def shift_img(img, disp_map):
    # Shifts an image according to a disparity map
    shifted_img = np.zeros((img.shape))

    for r in range(disp_map.shape[0]):
        for c in range(disp_map.shape[1]):
            shift = np.round(disp_map[r, c])
            if c-shift < 0:
                pass
            else:
                shifted_img[r,c-shift] = img[r, c]
    return shifted_img

def fill_holes(shifted_img, img):
    # If hole, replace with average of local neighborhood
    holes = np.array(np.where(shifted_img == 0)).T
    for index in range(holes.shape[0]):
        inds = holes[index,:]
        shifted_img[inds[0], inds[1]] = np.mean(img[inds[0]-5:inds[0]+5, inds[1]-5:inds[1]+5])
    return shifted_img

def synth_focus(img, disparities, windowsize, focused_disparity):
    # Synthetic refocusing algorithm

    disparities = np.round(disparities)

    disp_levels = range(int(np.min(disparities)), int(np.max(disparities)+1))
    refocused_img = np.zeros((img.shape[0], img.shape[1], 3, len(disp_levels)))

    for idx, disp in enumerate(disp_levels):
        # Mask of pixels in the iamge of the current disparity 
        #  value from the disparity map
        cur_mask = disparities == disp
        all_mask = np.zeros((cur_mask.shape[0], cur_mask.shape[1], 3))
        all_mask[:,:,0] = cur_mask
        all_mask[:,:,1] = cur_mask
        all_mask[:,:,2] = cur_mask
        all_mask = all_mask.astype(int)

        # Compute sigma of Gaussian blur according to disparity values
        sigma = 4* np.abs((disp - focused_disparity))*1./focused_disparity

        # Convolve the image with a Gaussian kernel with parameter sigma, 
        #  then multiple element-wise by the mask of the current disparity level pixels 
        if sigma == 0:
            refocused_img[:,:,:,idx] = np.multiply(img, all_mask)
        else:
            refocused_img[:,:,:,idx] = np.multiply(gauss_filt(img, sigma), all_mask)

    refocused_img = np.sum(refocused_img, 3)
    return refocused_img
            
def main():
    # Read in fused image and disparity map
    fused_img = np.asarray(Image.open("fused_img.png")) / 255.
    disp_map = np.load('results_dispmaps/disparity_SGBM_blocksize_filt_23.npy').astype(int)
    disp_map[disp_map < 0] = 0
    disp_map = np.nan_to_num(disp_map)

    disp_map_norm = disp_map*1.0 / np.max(disp_map)
    fused_img_norm = fused_img*1.0 / np.max(fused_img)

    # Shift the disparity map so that it is aligned with the fused image
    shifted_disparities = shift_img(disp_map, disp_map)
    shifted_disparities = np.nan_to_num(shifted_disparities)
    shifted_disp_norm = shifted_disparities*1.0 / np.max(shifted_disparities)

    # Fill holes in the shifted disparity map
    shifted_disparities = fill_holes(shifted_disparities, disp_map)
    shifted_disparities = np.round(shifted_disparities)
    
    windowsize= 7  # size of the Gaussian kerne used in blurring
    focused = 9    # Desired in-focus disparity level
    shifted_disparities = np.nan_to_num(shifted_disparities)
    fused_img = np.nan_to_num(fused_img)
    refocused_img = synth_focus(fused_img, shifted_disparities, windowsize, focused)

    plt.figure()
    plt.imshow(refocused_img)
    plt.show()

if __name__ == '__main__':
    main()
