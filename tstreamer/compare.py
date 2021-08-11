#!/usr/bin/python3
from skimage.metrics import structural_similarity as compare_ssim
import argparse
import imutils
import numpy as np
import cv2
from operator import *
from PIL import Image, ImageDraw

import glob


# load the two input images
im = "/mnt/nas_downloads/data/hassio/tstreamer/uvc_g3_pro_a_2021-07-25_17-11-07_unifiprotect_class_myx3m_75d2688b928942ceac94e188c34ffd20.jpg"
imageA = cv2.imread(im)

testimages = [
    "/mnt/nas_downloads/data/hassio/tstreamer/uvc_g3_pro_a_2021-07-25_15-51-08_unifiprotect_class_myx3m_4d9185a909db4abb81974994be1e0018.jpg",
]

testimages = glob.glob("/mnt/nas_downloads/data/hassio/tstreamer/uvc_g3_pro_a_*myx3*")

imatches = {}

for testimage in testimages:
    imatches[testimage] = {}

    imageB = cv2.imread(testimage)
    imageB = cv2.resize(imageB, (imageA.shape[1], imageA.shape[0]), interpolation = cv2.INTER_AREA)

    # convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    (scoressim, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    imatches[testimage]['SSIM'] = scoressim

    histogramA = cv2.calcHist([imageA], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    histogramB = cv2.calcHist([imageB], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    scorehist = cv2.compareHist(histogramA, histogramB, cv2.HISTCMP_INTERSECT)
    imatches[testimage]['Histogram'] = scoressim

    #  SURF
    fe = cv2.ORB_create()
    kpA, desc_a = fe.detectAndCompute(imageA, None)
    kpB, desc_b = fe.detectAndCompute(imageB, None)

    if desc_b is None:
        continue

    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    des1 = np.float32(desc_a)
    des2 = np.float32(desc_b) 
    matches = flann.knnMatch(des1, des2, k=2)
    ratio = 0.7
    similar_regions = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            similar_regions.append(m)
    if len(matches) == 0:
        scores = 0
    else:
        scores = len(similar_regions) / len(matches)
    imatches[testimage]['SIFT'] = scores

for key, value in imatches.items():
    print(key, ' : ', value)

print(im)