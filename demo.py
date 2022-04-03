import os
import glob
import torch
import torch.backends.cudnn
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
from PIL import Image
from utils import logger
from config import get_config
from model import AODnet,AODnet1
import cv2 
import numpy as np
import PIL

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = (os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def tensor_to_image(tensor):
    
    tensor = np.array(tensor.detach(), dtype=np.uint8)
    tensor = tensor*255
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)
@logger
def make_test_data(cfg, img_path_list, device,option):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize([411, 550]),
        torchvision.transforms.ToTensor()
    ])
    imgs = []
    for img_path in img_path_list:
        if(option==0):
            haze_image_name = Image.open(img_path)
            #haze_image_name=cv2.imread(img_path)
            #haze_image_name=cv2.cvtColor(haze_image_name,cv2.COLOR_BGR2YCrCb)
            #haze_image_name=cv2.imread(img_path)
            #haze_image_name=cv2.cvtColor(haze_image_name,cv2.COLOR_BGR2HSV)
            R, G, B = cv2.split(np.asarray(haze_image_name))
            output1_R = clahe.apply(R)
            output1_G = clahe.apply(G)
            output1_B = clahe.apply(B)
            image = cv2.merge((output1_R, output1_G, output1_B))
            image = Image.fromarray(np.uint8(image))
        elif(option==1):
            image = Image.open(img_path)
        elif(option == 2):
            haze_image_name=cv2.imread(img_path)
            haze_image_name=cv2.cvtColor(haze_image_name,cv2.COLOR_BGR2HSV)
            R, G, B = cv2.split(np.asarray(haze_image_name))
            output1_R = clahe.apply(R)
            output1_G = clahe.apply(G)
            output1_B = clahe.apply(B)
            image = cv2.merge((output1_R, output1_G, output1_B))
            image = Image.fromarray(np.uint8(image))
        elif(option == 3):
            haze_image_name=cv2.imread(img_path)
            haze_image_name=cv2.cvtColor(haze_image_name,cv2.COLOR_BGR2YCrCb)
            #haze_image_name=cv2.imread(img_path)
            #haze_image_name=cv2.cvtColor(haze_image_name,cv2.COLOR_BGR2HSV)
            R, G, B = cv2.split(np.asarray(haze_image_name))
            output1_R = clahe.apply(R)
            #output1_G = clahe.apply(G)
            #output1_B = clahe.apply(B)
            image = cv2.merge((output1_R, G, B))
            image = Image.fromarray(np.uint8(image))
            

        x = data_transform(image).unsqueeze(0)
        x = x.to(device)
        imgs.append(x)
    return imgs


@logger
def load_pretrain_network(cfg, device):
    net = AODnet().to(device)
    net1 = AODnet1().to(device)
    net2 = AODnet().to(device)
    net3 = AODnet().to(device)
    #net.load_state_dict(torch.load('D:/dataset/CLAHE_RGB.pkl')['state_dict'])
    net1.load_state_dict(torch.load('D:/dataset/AOD_4.pkl')['state_dict'])
    net2.load_state_dict(torch.load('D:/dataset/CLAHE_HSV3.pkl')['state_dict'])
    net3.load_state_dict(torch.load('D:/dataset/CLAHE_YCbCr6.pkl')['state_dict'])
    net.load_state_dict(torch.load('C:/Users/Aditya/AOD-Net-PyTorch/model/nets/AOD_4.pkl')['state_dict'])
    return net,net1,net2,net3


def main(cfg):
    # -------------------------------------------------------------------
    # basic config
    imgs=load_images_from_folder("D:/dataset/part2/")
    print(imgs[0])
    print(cfg)
    if cfg.gpu > -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # -------------------------------------------------------------------
    # load data
    # load network
    network,network1,network2,network3 = load_pretrain_network(cfg, device)
    # -------------------------------------------------------------------
    # set network weights
    # -------------------------------------------------------------------
    # start train
    print('Start eval')
    network.eval()
    network1.eval()
    network2.eval()
    network3.eval()
    for i in imgs:
        test_file_path = glob.glob(i)
        test_images = make_test_data(cfg, test_file_path, device,0)
        test_images1 = make_test_data(cfg, test_file_path, device,1)
        test_images2 = make_test_data(cfg, test_file_path, device,2)
        test_images3 = make_test_data(cfg, test_file_path, device,3)
    # -------------------------------------------------------------------
    
        for idx, im in enumerate(test_images):
            dehaze_image = network(im)
            dehaze_image1 = network1(test_images1[idx])
            dehaze_image2 = network2(test_images2[idx])
            dehaze_image3 = network3(test_images3[idx])
            #for k in range(0,1,1):
            #    dehaze_image = network(dehaze_image)
            
            #cv2.imshow("deh",tensor_to_image(dehaze_image.cpu()))
            #print(test_file_path[idx])
            #print(PSNR(im,dehaze_image))
            #torchvision.utils.save_image(torch.cat((test_images1[idx],dehaze_image1,dehaze_image), 0), "results1/" + test_file_path[idx].split("/")[-1])
            #torchvision.utils.save_image(test_images1[idx], "res2/" + (test_file_path[idx].split("/")[-1]))
            torchvision.utils.save_image(im, "CLAHE_Small/" +(test_file_path[idx].split("/")[-1]))
            #torchvision.utils.save_image(dehaze_image, "CLAHE_RGB/" +(test_file_path[idx].split("/")[-1]))
            #torchvision.utils.save_image(dehaze_image1, "AOD/" +(test_file_path[idx].split("/")[-1]))
            #torchvision.utils.save_image(dehaze_image2, "HSV/" +(test_file_path[idx].split("/")[-1]))
            #torchvision.utils.save_image(dehaze_image3, "YCbCr/" +(test_file_path[idx].split("/")[-1]))
            
    #cv2.imshow("deh","C:/Users/Aditya/AOD-Net-PyTorch/results/11.jpg")

if __name__ == '__main__':
    config_args, unparsed_args = get_config()
    main(config_args)
