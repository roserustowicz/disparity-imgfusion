'''
Implementation of Block Matching (BM) and Semi-Global Block Matching (SGBM)
using OpenCV. A post-processing step is applied to the SGBM result to fill
holes in the estimated disparity map.

Author: Rose Rustowicz, with reference to OpenCV webpages
Date: 16 March 2018
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt

#imgL = cv2.imread('tsukuba-imL.png',0) 
#imgR = cv2.imread('tsukuba-imR.png', 0) 
imgR = cv2.imread('../rectified_01_pan/07rectified_pan.png',0) 
imgL = cv2.imread('../rectified_00_rgb/07rectified_rgb.png',0)
imgR = cv2.resize(imgR, (960, 720))
imgL = cv2.resize(imgL, (960, 720))

# Block Matching
blockSizes = [5, 15, 25, 35, 45, 55]
for b_idx in range(len(blockSizes)):

    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=blockSizes[b_idx])
    disparity = stereo.compute(imgL, imgR)
    disparity = disparity/16.
    plt.imshow(disparity) #, 'gray')
    plt.title('Block Matching Disparity Map, blocksize: ' + str(blockSizes[b_idx]))
    plt.colorbar()
    plt.show()
    fname = 'disparity_BM_blocksize_' + str(blockSizes[b_idx])
    np.save(fname, disparity)

# Semi Global Block Matching 
blockSizes = [5, 11, 15, 23]
for b_idx in range(len(blockSizes)):
    stereo = cv2.StereoSGBM_create(numDisparities=64, blockSize=blockSizes[b_idx])
    right_matcher = cv2.ximgproc.createRightMatcher(stereo)
    
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.2
    visual_multiplier = 1.0
 
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    
    print('computing disparity...')
    displ = stereo.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr) 
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = filteredImg/16.
    filteredImg = np.uint8(filteredImg)
    plt.imshow(filteredImg)

    plt.title('Semi-Global Block Matching Disparity Map, blocksize: ' + str(blockSizes[b_idx]))
    plt.colorbar()
    plt.show()
    fname = 'disparity_SGBM_blocksize_filt_' + str(blockSizes[b_idx])
    np.save(fname, filteredImg)
