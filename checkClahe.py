import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
haze_image_name=cv2.imread('D:/use.jpg')
R,G,B = cv2.split(np.asarray(haze_image_name))
output1_R = clahe.apply(R)
output1_G = clahe.apply(G)
output1_B = clahe.apply(B)
haze_image_name = cv2.merge((output1_R, output1_G, output1_B))
cv2.imshow("Img",haze_image_name)
#cv2.imwrite("prat_out.jpg", haze_image_name)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([cv2.imread('D:/use.jpg')],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    #plt.xlim([0,256])
plt.show()
