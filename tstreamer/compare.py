#!/usr/bin/python3
from skimage.metrics import structural_similarity as compare_ssim
import argparse
import imutils
import cv2
from PIL import Image, ImageDraw

# load the two input images
imageA = cv2.imread("/mnt/nas_downloads/data/hassio/tstreamer_dev/uvc_g4_pro_946e_latest_object_car.jpg")
imageB = cv2.imread("/mnt/nas_downloads/data/hassio/tstreamer_dev/uvc_g4_pro_946e_2021-07-23_20-41-54_yolov5x-1280_object_car_da1cc43bad6c4517aae2e2644d803d3c.jpg")
imageB = cv2.resize(imageB, (imageA.shape[1], imageA.shape[0]), interpolation = cv2.INTER_AREA)
print(f"{imageA.shape[0]} {imageA.shape[1]}")
print(f"{imageB.shape[0]} {imageB.shape[1]}")

# convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

histogramA = cv2.calcHist([imageA], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
histogramB = cv2.calcHist([imageB], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
score = cv2.compareHist(histogramA, histogramB, cv2.HISTCMP_INTERSECT)
print("Histogram: {}".format(score))

#  SURF
surf = cv2.ORB_create()
kpA, desc_a = surf.detectAndCompute(imageA, None)
kpB, desc_b = surf.detectAndCompute(imageB, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(desc_a, desc_b)
similar_regions = [i for i in matches if i.distance < 70]
if len(matches) == 0:
    score = 0
else:
    score = len(similar_regions) / len(matches)
print("SURF: {}".format(score))