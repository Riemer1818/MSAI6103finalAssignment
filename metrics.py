import torch
import os, sys
import re 
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.uqi import  UniversalImageQualityIndex
from torchmetrics.image.vif import VisualInformationFidelity
import pandas as pd
import numpy as np
import cv2

# Learned perceptual image patch similarity
lpipsSqueeze= LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
lpipsAlex = LearnedPerceptualImagePatchSimilarity(net_type='alex')
lpipsVGG = LearnedPerceptualImagePatchSimilarity(net_type='vgg') # what vgg version are we using here?

# Peak-signal-to-noise-ratio
psnr = PeakSignalNoiseRatio()

# Structural similarity index measure
ssim = MultiScaleStructuralSimilarityIndexMeasure()

# Universal image quality index
uqi = UniversalImageQualityIndex()

# Visual information fidelity
vif = VisualInformationFidelity()

if __name__ == "__main__":
    # ask for system arguments

    # resultsDir = sys.argv[1] # path to the results directory
    resultsDir = "./results/facades_pix2pix/test_latest/images/"
    # outputDir = sys.argv[2] # path to the output directory
    outputDir = "./results/facades_pix2pix/test_latest/"

    pattern1 = re.compile(r'(\d+_(fake_B)\.png)')
    pattern2 = re.compile(r'(\d+_(real_B)\.png)')

    columns = ['fakeB', 'realB', 'lpipsSqueeze', 'lpipsAlex', 'lpipsVGG', 'psnr', 'ssim','uqi', 'vif']

    # get the list of files in the results directory
    fileList = os.listdir(resultsDir)
    fileList.sort()

    fakeB = []
    realB = []
    for file in fileList:
        # print(file)
        if pattern1.match(file):
            fakeB.append(file)
        elif pattern2.match(file):
            realB.append(file)
        else:
            continue

    dataframe = pd.DataFrame(columns=columns)
    for i in range(len(fakeB)):
        imgF = cv2.imread(os.path.join(resultsDir, fakeB[i]))
        imgR = cv2.imread(os.path.join(resultsDir, realB[i]))

        imgF = cv2.resize(imgF, (256, 256))
        imgR = cv2.resize(imgR, (256, 256))

        # Transpose dimensions to match the expected format [N, 3, H, W]
        imgF = np.transpose(imgF, (2, 0, 1))
        imgR = np.transpose(imgR, (2, 0, 1))

        imgF = np.expand_dims(imgF, axis=0)
        imgR = np.expand_dims(imgR, axis=0)

        # normalize 
        imgF = imgF / 255.0 * 2.0 - 1.0
        imgR = imgR / 255.0 * 2.0 - 1.0

        # Convert NumPy arrays to PyTorch tensors
        imgF = torch.from_numpy(imgF).float()
        imgR = torch.from_numpy(imgR).float()

        rowData = [fakeB[i], realB[i],
                    lpipsSqueeze(imgF, imgR).item(), 
                    lpipsAlex(imgF, imgR).item(), 
                    lpipsVGG(imgF, imgR).item(),
                    psnr(imgF, imgR).item(),
                    ssim(imgF, imgR).item(),
                    uqi(imgF, imgR).item(),
                    vif(imgF, imgR).item()]

        newRow = pd.DataFrame([rowData], columns=columns)
        dataframe = pd.concat([dataframe, newRow], ignore_index=True)

    print(dataframe)
    for column in columns[2:]:
        print(column, dataframe[column].mean(), dataframe[column].std())

    # save
    dataframe.to_csv(os.path.join(outputDir, 'metrics.csv'), index=False)

    # save mean and std in seperate file 
    mean = []
    std = []
    for column in columns[2:]:
        mean.append(dataframe[column].mean())
        std.append(dataframe[column].std())
    
    # save mean and std in seperate file with original columns as rows names
    dataframe = pd.DataFrame([mean, std], columns=columns[2:], index=['mean', 'std'])
    dataframe.to_csv(os.path.join(outputDir, 'metrics_mean_std.csv'))
