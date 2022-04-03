import cv2
import os
import numpy as np
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = (os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images
imgs=load_images_from_folder("C:/Users/Aditya/Pictures/compare/")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
for i in imgs:
    haze_image_name = cv2.imread(i)
    haze_image=cv2.cvtColor(haze_image_name,cv2.COLOR_BGR2Lab)
    R,G,B = cv2.split(np.asarray(haze_image_name))
    output1_R = clahe.apply(R)
    output1_G = clahe.apply(G)
    output1_B = clahe.apply(B)
    R,G,B = cv2.split(np.asarray(haze_image))
    output1_l = clahe.apply(R)
    output1_a = clahe.apply(G)
    output1_b = clahe.apply(B)
    haze_image_name = cv2.merge((output1_R, output1_G, output1_B))
    cv2.imshow("RGB",haze_image_name)
    haze_image = cv2.merge((output1_l, output1_a, output1_b))
    haze_image=cv2.cvtColor(haze_image,cv2.COLOR_Lab2BGR)
    cv2.imshow("lab",haze_image_name)
    cv2.waitKey(500)
