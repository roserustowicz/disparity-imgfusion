'''
Implementation of graph cuts with the Python GCO library. Cost functions 
are also implemented here. 

Author: Rose Rustowicz
Date: 16 March 2018
'''

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pygco import cut_simple
import cv2
from scipy.ndimage.filters import gaussian_filter as gauss_filt
import pdb

def get_MI_cost_array(img1, img2, max_disp, sigma):
    # Computes the mutual information cost array 
    
    img1 = (img1*255).astype(int)
    img2 = (img2*255).astype(int)
    
    img1_flat = np.asarray(img1.flatten()) 
    img2_flat = np.asarray(img2.flatten()) 

    # Calculate 2d histogram of intensities between img1 and img2, which 
    # represents the joint distribution of the two images 
    H, x_edges, y_edges = np.histogram2d(img1_flat, img2_flat, 256)
    
    # From H, calculate h_I1I2, which will serve as a look up table
    #  (LUT) when finding the mutual information cost array
    tmp = gauss_filt(H, sigma, 0)
    h_I1I2 = np.zeros(tmp.shape)
    non_zeros = tmp != 0
    h_I1I2[non_zeros] = -np.log(tmp[non_zeros])
    h_I1I2 = gauss_filt(h_I1I2, sigma, 0)
    h_I1I2 = gauss_filt(h_I1I2, sigma, 0)
 
    # Compute the mutual information cost array, using h_I1I2 as a LUT
    differences = []
    for disp in np.arange(max_disp):
        if disp == 0:
            tmp_im1 = img1
            tmp_im2 = img2

            # Index into h_I1I2 according to the intensities
            #  in img1 and img2 at this pixel location
            diff = h_I1I2[tmp_im1, tmp_im2]
        else:
            # Shift the images according to the disparity, then
            #  get the MI cost according to the pixel intensities
            #  and the h_I1I2 LUT
            tmp_im1 = img1[:,2*disp:] 
            tmp_im2 = img2[:,:-2*disp]
            diff = h_I1I2[tmp_im1, tmp_im2]
        if disp != max_disp - 1:
            diff = diff[:, max_disp - disp - 1:disp - max_disp + 1]
        differences.append(diff)
    result = np.dstack(differences).copy("C")
    
    return result

def census_transform(img1, img2):
    window_r = 9
    window_c = 7
    rows = img1.shape[0]
    cols = img1.shape[1]

    for r in range(window_r/2, rows - window_r/2):
        for c in range(window_c/2, cols - window_c/2):
            census = 0
            shift = 0
            # get local window to compare
            for m in range(r - window_r/2, r + window_r/2 + 1):
                for n in range(c - window_c/2, c + window_c/2 + 1):
                    if shift != window_r * window_c/2:
                        census1 <<= 1
                        if img1[m,n] < img1[r,c]:
                            bit1 = 1
                        else:
                            bit1 = 0
                        census1 = census1 + bit1
                        census2 <<= 1
                        if img2[m, n] < img2[r, c]:
                            bit2 = 1
                        else:
                            bit2 = 0
                    shift += 1
            img1[r,c] = census1
            img2[r,c] = census2

def hamming_cost(img1, img2):
    # Future Work
    pass

def shift_img(img, disp_map):
    shifted_img = np.zeros((img.shape))
    for r in range(disp_map.shape[0]):
        for c in range(disp_map.shape[1]):
            shift = np.round(disp_map[r, c])
            shifted_img[r,c+shift] = img[r, c]
    return shifted_img

def unaries_ssd(img1, img2, max_disp):
    # Computes the Sum of Squared Differences (SSD) cost array
    differences = []
    for disp in np.arange(max_disp):
        if disp == 0:
            diff = (img1 - img2) ** 2
        else:
            diff = (img1[:, 2 * disp:] - img2[:, :-2 * disp]) ** 2
        if disp != max_disp - 1:
            diff = diff[:, max_disp - disp - 1:disp - max_disp + 1]
        differences.append(diff)
    result = np.dstack(differences).copy("C")
    return result

def unaries_sad(img1, img2, max_disp):
    # Computes the Sum of Absolute Differences (SAD) cost array
    differences = []
    for disp in np.arange(max_disp):
        if disp == 0:
            diff = np.abs(img1 - img2) 
        else:
            diff = np.abs(img1[:, 2 * disp:] - img2[:, :-2 * disp]) 
        if disp != max_disp - 1:
            diff = diff[:, max_disp - disp - 1:disp - max_disp + 1]
        differences.append(diff)
    result = np.dstack(differences).copy("C")
    return result

def unaries_der_sad(img1, img2, max_disp, dtype):
    # Computes the SAD cost array on derivative images
    if dtype == 'dxdy_lap':
        img1, img2 = get_der_laplacian(img1, img2)
    elif dtype == 'dxdy_sobel':
        img1, img2 = get_der_sobel(img1, img2)
    elif dtype == 'dx_sobel':
        img1, img2 = get_der_sobel_dx(img1, img2)
    elif dtype == 'dy_sobel':
        img1, img2 = get_der_sobel_dy(img1, img2)

    result = unaries_sad(img1, img2, max_disp)
    return result

def unaries_der_ssd(img1, img2, max_disp, dtype):
    # Computes the SSD cost array on derivative images
    if dtype == 'dxdy_lap':
        img1, img2 = get_der_laplacian(img1, img2)
    elif dtype == 'dxdy_sobel':
        img1, img2 = get_der_sobel(img1, img2)
    elif dtype == 'dx_sobel':
        img1, img2 = get_der_sobel_dx(img1, img2)
    elif dtype == 'dy_sobel':
        img1, img2 = get_der_sobel_dy(img1, img2)

    result = unaries_ssd(img1, img2, max_disp)
    return result

def get_der_laplacian(img1, img2):
    # Computes images convolved with a Laplacian kernel
    laplacian_img1 = np.absolute(cv2.Laplacian(img1,cv2.CV_64F))
    laplacian_img2 = np.absolute(cv2.Laplacian(img2,cv2.CV_64F))
    return laplacian_img1, laplacian_img2

def get_der_sobel(img1, img2):
    # Computes images convolved with a Sobel kernel
    sobelx_img1 = cv2.Sobel(img1,cv2.CV_64F,1,0,ksize=5)
    sobely_img1 = cv2.Sobel(img1,cv2.CV_64F,0,1,ksize=5)
    sobel_img1 = sobelx_img1 + sobely_img1
    sobelx_img2 = cv2.Sobel(img2,cv2.CV_64F,1,0,ksize=5)
    sobely_img2 = cv2.Sobel(img2,cv2.CV_64F,0,1,ksize=5)
    sobel_img2 = sobelx_img2 + sobely_img2
    return sobel_img1, sobel_img2
    
def get_der_sobel_dy(img1, img2):
    # Computes images convolved with a dy Sobel kernel
    sobely_img1 = cv2.Sobel(img1,cv2.CV_64F,0,1,ksize=5)
    sobely_img2 = cv2.Sobel(img2,cv2.CV_64F,0,1,ksize=5)
    return sobely_img1, sobely_img2

def get_der_sobel_dx(img1, img2):
    # Computes images convolved with a dx Sobel kernel
    sobelx_img1 = cv2.Sobel(img1,cv2.CV_64F,1,0,ksize=5)
    sobelx_img2 = cv2.Sobel(img2,cv2.CV_64F,1,0,ksize=5)
    return sobelx_img1, sobelx_img2

def original_example(img1, img2, max_disp):
    # Code was based on this example
    unaries = (unaries_ssd(img1, img2, max_disp) * 100).astype(np.int32)
    n_disps = unaries.shape[2]

    newshape = unaries.shape[:2]
    potts_cut1 = cut_simple(unaries, -5 * np.eye(n_disps, dtype=np.int32))
    potts_cut2 = cut_simple(unaries, -5 * np.eye(n_disps, dtype=np.int32), n_iter = 10)
    x, y = np.ogrid[:n_disps, :n_disps]
    one_d_topology = np.abs(x - y).astype(np.int32).copy("C")

    one_d_cut1 = cut_simple(unaries, 5 * one_d_topology)
    one_d_cut2 = cut_simple(unaries, 5 * one_d_topology, n_iter = 10)
    return one_d_cut1, one_d_cut2, potts_cut1, potts_cut2

def some_cut(unaries, binaries, K, n_iter):
    # Performs the graph cut
    n_disps = unaries.shape[2]
    newshape = unaries.shape[:2]
    cut = cut_simple(unaries, K * binaries, n_iter)
    return cut

def main():
    # Set constants
    max_disp = 64 # Maximum disparities
    sigma = 1.5   # Sigma used in Gaussian blurring for MI cost array 
    K = 3         # Constant that controls smoothness strength
    
    img2 = np.asarray(Image.open("../rectified_00_rgb/07rectified_rgb.png")) / 255.
    img1 = np.asarray(Image.open("../rectified_01_pan/07rectified_pan.png")) / 255.

    img2 = np.mean(img2,2)
    img1 = img1[:,:,1]
    
    img1 = cv2.resize(img1, (960, 720)) 
    img2 = cv2.resize(img2, (960, 720)) 

    tmp2 = img2
    img2 = img1
    img1 = tmp2


    # Define UNARY costs, data terms
    #dataterm_ssd_dxdy_lap = (unaries_der_ssd(img1, img2, max_disp, 'dxdy_lap')*100).astype(np.int32)
    #dataterm_ssd_dxdy_sobel = (unaries_der_ssd(img1, img2, max_disp, 'dxdy_sobel')*100).astype(np.int32)
    #dataterm_ssd_dx_sobel = (unaries_der_ssd(img1, img2, max_disp, 'dx_sobel')*100).astype(np.int32)
    #dataterm_ssd_dy_sobel = (unaries_der_ssd(img1, img2, max_disp, 'dy_sobel')*100).astype(np.int32)
    #dataterm_ssd = (unaries_ssd(img1, img2, max_disp)*100).astype(np.int32)
    #dataterm_sad_dxdy_lap = (unaries_der_sad(img1, img2, max_disp, 'dxdy_lap')*100).astype(np.int32)
    #dataterm_sad_dxdy_sobel = (unaries_der_sad(img1, img2, max_disp, 'dxdy_sobel')*100).astype(np.int32)
    #dataterm_sad_dx_sobel = (unaries_der_sad(img1, img2, max_disp, 'dx_sobel')*100).astype(np.int32)
    #dataterm_sad_dy_sobel = (unaries_der_sad(img1, img2, max_disp, 'dy_sobel')*100).astype(np.int32)
    dataterm_sad = (unaries_sad(img1, img2, max_disp)*100).astype(np.int32)

    # Normalize MI between 0 - 1
    dataterm_mi = get_MI_cost_array(img1, img2, max_disp, sigma)
    dataterm_mi = (dataterm_mi - np.min(dataterm_mi))/(np.max(dataterm_mi) - np.min(dataterm_mi))
    dataterm_mi = (dataterm_mi*100).astype(np.int32)
    
    # Define BINARY costs, smoothness terms
    #potts_V = -K * np.eye(max_disp, dtype=np.int32)    

    x, y = np.ogrid[:max_disp, :max_disp]
    onedtopo_V = np.abs(x - y).astype(np.int32).copy("C")

    #custom_V = np.zeros((max_disp, max_disp))
    #for r in range(max_disp):
    #    for c in range(max_disp):
    #        if np.abs(r - c) == 1:
    #        #if np.abs(r - c) < 1:
    #            custom_V[r, c] = 1 
    #custom_V = custom_V.astype(np.int32)

    # Create results from different calls to the different functions
    unaries = [dataterm_sad, dataterm_mi]
    binaries = [onedtopo_V] 

    count = 0
    for u_idx in range(len(unaries)):
        for b_idx in range(len(binaries)):
            for iter_idx in [1, 5]: #3, 5, 7]:
                cut = some_cut(unaries[u_idx], binaries[b_idx], K, iter_idx)
                plt.figure()
                plt.imshow(cut) 
                plt.colorbar()
                plt.title(str(count))
                plt.show(block=False)
                plt.pause(0.0001)

                fname =  str(count) + '_disparity_map'
                np.save(fname, cut)
                count += 1
    plt.show()

main()
