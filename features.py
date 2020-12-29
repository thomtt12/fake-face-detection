import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from typing import Union
import warnings
from skimage.feature import local_binary_pattern

'''
Implementing the features used in the paper:
Javier Galbally and Sébastien Marcel. Face anti-spoofing based on general image quality assessment. In ICPR ’14 Proceedings of the 2014 22nd International Conference on Pattern Recognition, pages 1173–1178, 2014.
'''

# Pixel Difference Measures
def MeanSquareError(I: Union[list, np.ndarray] , I_m: Union[list, np.ndarray]):
    '''
    parameters:
    I: np.ndarray or list, image
    I_m: np.ndarray or list, image I after Gaussian filter
    return:
    mean_squared_error between 2 array
    '''
    I ,I_m = np.array(I), np.array(I_m)
    assert I.shape == I_m.shape
    
    return mean_squared_error(np.ravel(I), np.ravel(I_m))
    
def PeakSignal2NoiseRatio(I: Union[list, np.ndarray] , I_m: Union[list, np.ndarray]):
    '''
    parameters:
    I: np.ndarray or list, image
    I_m: np.ndarray or list, image I after Gaussian filter
    return:
    Peak Signal_to_noise ratio between 2 array
    '''
    #print('peak')
    I ,I_m = np.array(I), np.array(I_m)

    assert I.shape == I_m.shape
    
    if(MeanSquareError(I, I_m) == 0):
        return 100
    
    psnr = 10 * np.log10(np.max(np.power(I,2)) / MeanSquareError(I, I_m)) 
    return psnr 

def Signal2NoiseRatio(I: Union[list, np.ndarray] , I_m: Union[list, np.ndarray]):
    '''
    parameters:
    I: np.ndarray or list, image
    I_m: np.ndarray or list, image I after Gaussian filter
    return:
    Signal_to_noise ratio between 2 array
    '''
    #print('sig2noi')
    I ,I_m = np.array(I), np.array(I_m)

    assert I.shape == I_m.shape
    
    if(MeanSquareError(I, I_m) == 0):
        return 100
    
    snr = 10 * np.log10(np.sum(np.power(I, 2)) / (I.shape[0] * I.shape[1] * MeanSquareError(I, I_m)))
    return snr

def StructuralContent(I: Union[list, np.ndarray] , I_m: Union[list, np.ndarray]):
    '''
    parameters:
    I: np.ndarray or list, image
    I_m: np.ndarray or list, image I after Gaussian filter
    return:
    Structural Content between 2 array
    '''
    #print('struct')
    I ,I_m = np.array(I), np.array(I_m)

    assert I.shape == I_m.shape
        
    sc = np.sum(np.power(I, 2)) / np.sum(np.power(I_m, 2))
    return sc
    
def MaxDiff(I: Union[list, np.ndarray] , I_m: Union[list, np.ndarray]):
    '''
    parameters:
    I: np.ndarray or list, image
    I_m: np.ndarray or list, image I after Gaussian filter
    return:
    Maximum difference of value in the same pixel
    '''
    #print('maxdiff')
    I ,I_m = np.array(I), np.array(I_m)

    assert I.shape == I_m.shape
        
    max_diff = np.max(np.abs(I - I_m))
    return max_diff

def AvgDiff(I: Union[list, np.ndarray] , I_m: Union[list, np.ndarray]):
    '''
    parameters:
    I: np.ndarray or list, image
    I_m: np.ndarray or list, image I after Gaussian filter
    return:
    Average difference of all values of pixels
    '''
    #print('avgdiff')
    I ,I_m = np.array(I), np.array(I_m)

    assert I.shape == I_m.shape
        
    avg_diff = np.sum(I - I_m) / (I.shape[0] * I.shape[1])
    return avg_diff

def NormalizeAbsError(I: Union[list, np.ndarray] , I_m: Union[list, np.ndarray]):
    '''
    parameters:
    I: np.ndarray or list, image
    I_m: np.ndarray or list, image I after Gaussian filter
    return:
    Normalized Absolute Error between 2 array
    '''
    #print('normabserr')
    I ,I_m = np.array(I), np.array(I_m)
    
    assert I.shape == I_m.shape
    
    nae = np.sum(np.abs(I - I_m)) / np.sum(np.abs(I))
    
    return nae
    
def rAveragedMaxDiff(I: Union[list, np.ndarray] , I_m: Union[list, np.ndarray], r: int = 10):
    '''
    parameters:
    I: np.ndarray or list, image
    I_m: np.ndarray or list, image I after Gaussian filter
    r: int, index of the pixel difference ranking
    return:
    max r is defined as the r-highest pixel difference between two images. For the paper's implementation, R = 10.
    '''
    #print('rAvgmaxdiff')
    I ,I_m = np.array(I), np.array(I_m)
    
    assert I.shape == I_m.shape
    
    if(r == 0):
        #Assuming you want to implement the max difference
        r = 1
    
    if(r > I.shape[0] * I.shape[1]):
        warnings.warn(f"Warning: r={r} is larger than number of total pixels")
        r = I.shape[0] * I.shape[1]
        
    maxes = np.ravel(I) - np.ravel(I_m)
    maxes = np.sort(maxes)[::-1]
    
    ramd = 1/r * np.sum(maxes[:r]) 
    
    return ramd

def _Laplacian_h_(I, row, col):
    I = I.astype('float64')
    h_I = I[row + 1, col] + I[row - 1 , col]  
    + I[row, col + 1] + I[row , col - 1] - 4*I[row, col]
    return h_I

def Laplacian_MeanSquaredError(I: Union[list, np.ndarray] , I_m: Union[list, np.ndarray]):
    '''
    parameters:
    I: np.ndarray or list, image
    I_m: np.ndarray or list, image I after Gaussian filter
    return:
    Laplacian Mean Squared Error
    '''
    #print('laplacian')
    I ,I_m = np.array(I), np.array(I_m)
    
    assert I.shape == I_m.shape
    
    numerator = 0
    denominator = 0
    for row in range(1, I.shape[0] - 1):
        for col in range(2, I.shape[1] - 1):
            h_I = _Laplacian_h_(I, row, col)
            h_I_m = _Laplacian_h_(I_m, row, col)
            numerator += np.power((h_I - h_I_m), 2)
            denominator += np.power(h_I, 2)
    
    return numerator / np.float(denominator)

#Correlation-based Measures
def NormalizeCrossCorrelation(I: Union[list, np.ndarray] , I_m: Union[list, np.ndarray]):
    '''
    parameters:
    I: np.ndarray or list, image
    I_m: np.ndarray or list, image I after Gaussian filter
    return:
    Laplacian Mean Squared Error
    '''
    I ,I_m = np.array(I), np.array(I_m)
    
    assert I.shape == I_m.shape
                              
    nxc = np.sum(np.multiply(I, I_m)) / np.sum(np.power(I_m, 2))
    
    return nxc
                            
def _AngleBetweenPixelVector_(I, I_m, row, col):
    
    if(np.linalg.norm(I[row, col, :]) * np.linalg.norm(I_m[row, col, :]) == 0): #Pixel vectors that are [0. 0. 0.]
        return 0
        
    cos_alpha = np.dot(I[row, col, :], I_m[row, col, :]) / (np.linalg.norm(I[row, col, :]) * np.linalg.norm(I_m[row, col, :]))
        
    if(np.abs(cos_alpha - 1) < 0.00001): #Control for python's capability
        cos_alpha = 1
        
    assert cos_alpha <= 1
    assert cos_alpha >= -1
    
    angle = np.arccos(cos_alpha)
    return (2/np.pi) * angle

def MeanAngleSimilarity(I: Union[list, np.ndarray] , I_m: Union[list, np.ndarray]):
    '''
    parameters:
    I: np.ndarray or list, image, 3 channels
    I_m: np.ndarray or list, image I after Gaussian filter, 3 channels
    return:
    Mean Angle Similarity
    '''
    #print('meananglesimi')
    I ,I_m = np.array(I), np.array(I_m)
    
    assert I.shape == I_m.shape
    #Apply on pictures with 3 channels
    assert (len(I.shape) >= 3)
    
    MAS = 0
    for row in range(I.shape[0]):
        for col in range(I_m.shape[1]):
            MAS += _AngleBetweenPixelVector_(I, I_m, row, col)
    MAS = 1 - 1/(I.shape[0] * I.shape[1]) * MAS     
    
    return MAS
    
def MeanAngleMagnitudeSimilarity(I: Union[list, np.ndarray] , I_m: Union[list, np.ndarray]):
    '''
    parameters:
    I: np.ndarray or list, image
    I_m: np.ndarray or list, image I after Gaussian filter
    return:
    Mean Angle Magnitude Similarity
    '''
    #print('meananglemagsimi')
    I ,I_m = np.array(I), np.array(I_m)
    
    assert I.shape == I_m.shape
    #Apply on pictures with 3 channels
    assert (len(I.shape) >= 3)
    
    MAMS = 0
    for row in range(I.shape[0]):
        for col in range(I.shape[1]):
            MAMS += 1 - (1-_AngleBetweenPixelVector_(I, I_m, row, col))*(1 - np.linalg.norm(I[row, col,:] - I_m[row, col,:])/255)
            
    MAMS = (1/(I.shape[0] * I.shape[1])) * MAMS
    
    return MAMS

#Edge-based Measures
def _to_binary_edge_(I):
    #Transform any matrix to binary matrix.
    #With 1 for values larger than mean, 0 for values smaller than mean
    threshold = np.mean(I)
    I[I < threshold] = 0.
    I[I >= threshold]= 1.
    return I

def _to_sobel_binary_edge_map_(I, ksize = 3):
    #Transform any matrix to its binary edge map.
    #cv2.Sobel already include Gaussian filtering
    sobelx = cv2.Sobel(I,cv2.CV_64F,1,0,ksize)
    sobely = cv2.Sobel(I,cv2.CV_64F,0,1,ksize=5)
    
    sobelxy = np.sqrt(np.power(sobelx, 2) + np.power(sobely, 2))
    
    return _to_binary_edge_(sobelxy)
    
def TotalEdgeDifference(I: Union[list, np.ndarray] , I_m: Union[list, np.ndarray]):
    '''
    parameters:
    I: np.ndarray or list, image
    I_m: np.ndarray or list, image I after Gaussian filter
    return:
    Total Edge Difference
    '''
    #print('edge')
    I ,I_m = np.array(I), np.array(I_m)
    
    assert I.shape == I_m.shape
    
    sobel_original = _to_sobel_binary_edge_map_(I)
    sobel_filtered = _to_sobel_binary_edge_map_(I_m)
    
    return np.sum(sobel_original - sobel_filtered) / (I.shape[0] * I.shape[1])

def _num_corner_(I: np.ndarray, 
                blockSize:int = 2,
                ksize:int = 3,
                k:float = 0.04,
                threshold_ratio:float = 0.01):
    '''
    I: matrix
    blockSize: Neighborhood size 
    ksize:Aperture parameter for the Sobel operator.
    k:Harris detector free parameter 
    '''
    #Return the number of corners the image has
    result = cv2.cornerHarris(I,blockSize,ksize,k)
    num_corner = len([x for x in result.ravel() if x >= threshold_ratio * np.max(result)])
    
    return num_corner
    
def TotalCornerDifference(I: Union[list, np.ndarray] , I_m: Union[list, np.ndarray]):
    '''
    parameters:
    I: np.ndarray or list, image
    I_m: np.ndarray or list, image I after Gaussian filter
    return:
    Total Corner Difference
    '''
    #print('corner')
    I ,I_m = np.array(I), np.array(I_m)
    
    assert I.shape == I_m.shape
    assert len(I.shape) == 2
    
    return np.abs(_num_corner_(I) - _num_corner_(I_m)) / max(_num_corner_(I), _num_corner_(I_m))
    
'''
Implementing the features used in the paper:
Zinelabidine Boulkenafet, Jukka Komulainen, and Abdenour Hadid. Face anti-spoofing based on
color texture analysis. In 2015 IEEE international conference on image processing (ICIP), pages 2636–2640. IEEE, 2015
'''
def LBPHistogram(image,P=8,R=1,method = 'nri_uniform'):
    '''
    image: shape is N*M 
    P: number of points considered per pixel
    R: radius of neighborhood
    '''
    lbp = local_binary_pattern(image, P,R, method) # lbp.shape is equal image.shape
    max_bins = int(lbp.max() + 1) # max_bins is related P
    hist,_= np.histogram(lbp,  normed=True, bins=max_bins, range=(0, max_bins))
    return hist