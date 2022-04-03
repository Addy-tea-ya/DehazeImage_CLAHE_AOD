from PIL import Image
import os

arr1=['4749_0.8_0.2.jpg',
'4757_0.8_0.2.jpg',
'4800_0.95_0.2.jpg',
'4830_0.8_0.2.jpg',
'4864_1_0.16.jpg',
'4874_1_0.2.jpg',
'5001_0.95_0.16.jpg',
'5086_1_0.2.jpg',
'5366_0.9_0.2.jpg',
'5352_1_0.2.jpg']

arr2=['4747_0.9_0.2.jpg',
      '5154_0.85_0.2.jpg',
      '5158_1_0.12.jpg'
    ]
arr3=['4771_0.9_0.2.jpg',
      '4803_0.9_0.16.jpg',
      '4807_0.95_0.2.jpg',
      '4836_0.85_0.2.jpg',
      '4866_1_0.2.jpg',
      '4869_0.8_0.2.jpg',
      '4871_0.9_0.2.jpg',
      '4999_0.95_0.2.jpg'
    ]
'''
os.mkdir('C:/Users/Aditya/Documents/paper/GroundTruth')
os.mkdir('C:/Users/Aditya/Documents/paper/HazyInput')
os.mkdir('C:/Users/Aditya/Documents/paper/Proposed(CLAHE_RGB_AOD)')
os.mkdir('C:/Users/Aditya/Documents/paper/Proposed1(CLAHE_YCbCr_AOD)')
os.mkdir('C:/Users/Aditya/Documents/paper/AOD_Net')
os.mkdir('C:/Users/Aditya/Documents/paper/FFA_Net')
os.mkdir('C:/Users/Aditya/Documents/paper/PFF_Net')
os.mkdir('C:/Users/Aditya/Documents/paper/Gman')
'''
'''
os.mkdir('C:/Users/Aditya/Documents/paper/GroundTruth1')
os.mkdir('C:/Users/Aditya/Documents/paper/HazyInput1')
os.mkdir('C:/Users/Aditya/Documents/paper/Proposed(CLAHE_RGB_AOD)1')
'''
os.mkdir('C:/Users/Aditya/Documents/paper/GroundTruth_Data')
os.mkdir('C:/Users/Aditya/Documents/paper/HazyInput1_Data')
'''
for i in arr1:
    
    imge=(i.split("_"))[0]+'.jpg'
    print(imge)
    img=Image.open('D:/dataset/clear2/'+imge)
    img = img.resize((550,411))
    img.save('C:/Users/Aditya/Documents/paper/GroundTruth/'+imge)

    img=Image.open('D:/dataset/part2/'+i)
    img = img.resize((550,411))
    img.save('C:/Users/Aditya/Documents/paper/HazyInput/'+i)

    img=Image.open('C:/Users/Aditya/AOD-Net-PyTorch/CLAHE_RGB/'+i)
    img = img.resize((550,411))
    img.save('C:/Users/Aditya/Documents/paper/Proposed(CLAHE_RGB_AOD)/'+i)

    img=Image.open('C:/Users/Aditya/AOD-Net-PyTorch/YCbCr/'+i)
    img = img.resize((550,411))
    img.save('C:/Users/Aditya/Documents/paper/Proposed1(CLAHE_YCbCr_AOD)/'+i)

    img=Image.open('C:/Users/Aditya/AOD-Net-PyTorch/AOD/'+i)
    img = img.resize((550,411))
    img.save('C:/Users/Aditya/Documents/paper/AOD_Net/'+i)

    img=Image.open('C:/Users/Aditya/FFA-Net/net/pred_FFA_ots/'+i)
    img = img.resize((550,411))
    img.save('C:/Users/Aditya/Documents/paper/FFA_Net/'+i)

    img=Image.open('C:/Users/Aditya/PFFNet/output/'+i)
    img = img.resize((550,411))
    img.save('C:/Users/Aditya/Documents/paper/PFF_Net/'+i)

    img=Image.open('C:/Users/Aditya/Image-Dehazing-using-GMAN-net/ResultsOp/'+i)
    img = img.resize((550,411))
    img.save('C:/Users/Aditya/Documents/paper/Gman/'+i)
'''
for i in arr3:
    imge=(i.split("_"))[0]+'.jpg'
    print(imge)
    img=Image.open('D:/dataset/clear2/'+imge)
    img = img.resize((550,411))
    img.save('C:/Users/Aditya/Documents/paper/GroundTruth_Data/'+imge)

    img=Image.open('D:/dataset/part2/'+i)
    img = img.resize((550,411))
    img.save('C:/Users/Aditya/Documents/paper/HazyInput1_Data/'+i)
    '''
    img=Image.open('C:/Users/Aditya/AOD-Net-PyTorch/CLAHE_RGB/'+i)
    img = img.resize((550,411))
    img.save('C:/Users/Aditya/Documents/paper/Proposed(CLAHE_RGB_AOD)1/'+i)

    '''
    

    

    

    
    
            
        
