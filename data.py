import os
import torch
from PIL import Image
import glob
import random
from torchvision import transforms
import cv2
import numpy as np
class HazeDataset(torch.utils.data.Dataset):
    def __init__(self, ori_root, haze_root, transforms,diff):
        self.haze_root = haze_root
        self.ori_root = ori_root
        self.image_name_list = glob.glob(os.path.join(self.haze_root, '*.jpg'))
        self.matching_dict = {}
        self.file_list = []
        self.get_image_pair_list(diff)
        self.transforms = transforms
        print("Total data examples:", len(self.file_list))

    def __getitem__(self, item):
        """
        :param item:
        :return: haze_img, ori_img
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        ori_image_name, haze_image_name = self.file_list[item]
        #ori_image_name = clahe.apply(np.asarray(Image.open(ori_image_name)))
        #haze_image_name = clahe.apply(np.asarray(Image.open(haze_image_name)))
        ori_image_name=Image.open(ori_image_name)
        haze_image_name=Image.open(haze_image_name)
        #haze_image_name=cv2.imread(haze_image_name)
        #haze_image_name=cv2.cvtColor(haze_image_name,cv2.COLOR_BGR2YCrCb)
        '''
        R, G, B = cv2.split(np.asarray(ori_image_name))
        output1_R = clahe.apply(R)
        output1_G = clahe.apply(G)
        output1_B = clahe.apply(B)
        ori_image_name = cv2.merge((output1_R, output1_G, output1_B))
        ori_image_name = Image.fromarray(np.uint8(ori_image_name))
        '''
        R,G,B = cv2.split(np.asarray(haze_image_name))
        output1_R = clahe.apply(R)
        output1_G = clahe.apply(G)
        output1_B = clahe.apply(B)
        haze_image_name = cv2.merge((output1_R, output1_G, output1_B))
        #ori_image_name=ori_image_name.resize([550,413])
        #haze_image_name=haze_image_name.resize([550,413])
        #cv2.imshow("Haze",haze_image_name)
        haze_image_name = Image.fromarray(np.uint8(haze_image_name))
        
        ori_image = self.transforms(ori_image_name)
        haze_image = self.transforms(haze_image_name)
        return ori_image, haze_image

    def __len__(self):
        return len(self.file_list)

    def get_image_pair_list(self,diff):
        if(diff=="NYU"):
            for image in self.image_name_list:
                image = (image.split("/")[-1])[5:]
                key = image.split("_")[0] + "_" + image.split("_")[1] + ".jpg"
               
                if key in self.matching_dict.keys():
                    self.matching_dict[key].append(image)
                else:
                    self.matching_dict[key] = []
                    self.matching_dict[key].append(image)

            for key in list(self.matching_dict.keys()):
                for hazy_image in self.matching_dict[key]:
                    #print(key,hazy_image)
                    self.file_list.append([os.path.join(self.ori_root, key), os.path.join(self.haze_root, hazy_image)])
                    #print(os.path.join(self.ori_root, key), os.path.join(self.haze_root, hazy_image))


            random.shuffle(self.file_list)
            
        elif(diff=="RESIDE"):   
            for image in self.image_name_list:
                
                image = (image.split("/")[-1])[5:]
                key = image.split("_")[0] + ".jpg"
               
                if key in self.matching_dict.keys():
                    self.matching_dict[key].append(image)
                else:
                    self.matching_dict[key] = []
                    self.matching_dict[key].append(image)

            for key in list(self.matching_dict.keys()):
                for hazy_image in self.matching_dict[key]:
                    #print(key,hazy_image)
                    self.file_list.append([os.path.join(self.ori_root, key), os.path.join(self.haze_root, hazy_image)])
                    #print(os.path.join(self.ori_root, key),os.path.join(self.haze_root, hazy_image))
            random.shuffle(self.file_list)
            
             
        

