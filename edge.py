import cv2
import numpy as np
import os
import torch



path = r'D:\Datasets\DhMurals-inpainting-dataset\DhMurals_Blind_inpainting_strokes\train\input_merged_01_places2'
save_gray = r'D:\Datasets\DhMurals-inpainting-dataset\DhMurals_Blind_inpainting_strokes\train\input_merged_01_places2_gray'
save_path =r'D:\Datasets\DhMurals-inpainting-dataset\DhMurals_Blind_inpainting_strokes\train\input_merged_01_places2_edge'
file_list = os.listdir(path)
for i,file_name in enumerate(file_list):
    # 读取图像
    image = cv2.imread(os.path.join(path,file_name), cv2.IMREAD_GRAYSCALE)
    # # 使用Canny算法进行边缘检测
    # edges = cv2.Canny(image, threshold1=150, threshold2=200)
### sobel
    x = cv2.Sobel(image,cv2.CV_16S,2,0,ksize=3)
    y = cv2.Sobel(image, cv2.CV_16S, 0, 2,ksize=3)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Sobel = cv2.addWeighted(absX,1,absY,1,0)
    # edges = cv2.Canny(Sobel, threshold1=200, threshold2=250)
    cv2.imwrite(os.path.join(save_gray,file_name), image)
    cv2.imwrite(os.path.join(save_path,file_name), Sobel)