import numpy as np
import cv2
import glob as glob
import matplotlib.pyplot as plt
from itertools import product
from sklearn.metrics import mean_squared_error
from typing import Union
import warnings
from features import *
from evaluate import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


class ImageQualityAssessmentPAD():
    def __init__(self):
        pass
    
    def _createGalballyMarcelFeatureVector_(self, img, GKernelSize = (3,3), sigmaX = 0.5, sigmaY = 0.5):
        '''
        Outputing an array of features used in the paper:
        Javier Galbally and Sébastien Marcel. Face anti-spoofing based on general image quality assessment. 
        In ICPR ’14 Proceedings of the 2014 22nd International Conference on Pattern Recognition, 
        pages 1173–1178, 2014.
        '''
        img_gray = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img, GKernelSize, sigmaX, sigmaY)
        img_gray_blur = cv2.GaussianBlur(img_gray, GKernelSize, sigmaX, sigmaY)
        
        features = []
        
        #Pixel-based Measurements
        features.append(MeanSquareError(img_gray, img_gray_blur))
        features.append(PeakSignal2NoiseRatio(img_gray, img_gray_blur))
        features.append(Signal2NoiseRatio(img_gray, img_gray_blur))
        features.append(StructuralContent(img_gray, img_gray_blur))
        features.append(MaxDiff(img_gray, img_gray_blur))
        features.append(AvgDiff(img_gray, img_gray_blur))
        features.append(NormalizeAbsError(img_gray, img_gray_blur))
        features.append(rAveragedMaxDiff(img_gray, img_gray_blur))
#         features.append(Laplacian_MeanSquaredError(img_gray, img_gray_blur))

        #Correlation-based Measurements
        features.append(NormalizeCrossCorrelation(img, img_blur))
#         features.append(MeanAngleSimilarity(img, img_blur))
#         features.append(MeanAngleMagnitudeSimilarity(img, img_blur))
        
        #Edge-based Measurements
        features.append(TotalEdgeDifference(img_gray, img_gray_blur))
        features.append(TotalCornerDifference(img_gray, img_gray_blur))
        
        assert len(features) == 11
        
        return features
    
    def _createBoulkenafetHSVFeatureVector_(self, img):
        HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h = LBPHistogram(HSV[:,:,0]) # y channel
        s = LBPHistogram(HSV[:,:,1]) # cb channel
        v = LBPHistogram(HSV[:,:,2]) # cr channel
        feature = np.concatenate((h, s, v))
        return feature
    
    def _createBoulkenafetYCrCbFeatureVector_(self, img):
        YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y_h = LBPHistogram(YCrCb[:,:,0]) # y channel
        cb_h = LBPHistogram(YCrCb[:,:,1]) # cb channel
        cr_h = LBPHistogram(YCrCb[:,:,2]) # cr channel
        feature = np.concatenate((y_h,cb_h,cr_h))
        return feature
    
    def DataGenerator(self, 
                      x_train: list,
                      implementation = 'GalballyMarcel'):
        #Output a features matrix (num_samples, num_features)
        X = []
        
#         i = 0
        for file_path in x_train:
            img = plt.imread(file_path)
#             print(i)
            if(implementation == 'GalballyMarcel'):
                X.append(self._createGalballyMarcelFeatureVector_(img))
            elif(implementation == 'BoulkenafetHSV'):
                X.append(self._createBoulkenafetHSVFeatureVector_(img))
            elif(implementation == 'BoulkenafetYCrCb'):
                X.append(self._createBoulkenafetYCrCbFeatureVector_(img))
#             i += 1
                
        X = np.array(X)
        assert (X.shape[0] == len(x_train))
        
        return X
        
    def fit(self, 
            x_train, 
            y_train: Union[list, np.ndarray],
            typeOfClassifier = 'LDA'):
        
        #Load data
        if(type(x_train[0]) == str):
            X = self.DataGenerator(x_train)
        else:
            X = x_train
        
        #Load label
        y = np.array(y_train)
        
        assert(y.shape[0] == X.shape[0])
        
        if(typeOfClassifier == 'LDA'):
            clf = LinearDiscriminantAnalysis()
        elif(typeOfClassifier == 'SVM'):
            clf = SVC(kernel='rbf', C=1e3, gamma=0.5, class_weight='balanced', probability=True)
        elif(typeOfClassifier == 'LR'):
            clf = LogisticRegression()
        
        clf.fit(X, y)
        
        self.clf = clf
        return clf
        
    def predict(self,
                x_test):
        
        #Load data
        if(type(x_test[0]) == str):
            X = DataGenerator(x_test)
        else:
            X = x_test
        
        y_pred = self.clf.predict(x_test)
        return y_pred
        