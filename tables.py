import os
import glob
from tabulate import tabulate
from pandas import DataFrame
from timeit import timeit
import matplotlib.pyplot as plt
def arr(arg):
    f = open(arg, 'r+')
    #r1 = [float(line[:-2]) for line in f.readlines()]
    r1 = [float(line.strip()) for line in f.readlines()]
    f.close()
    return r1
arrP=[]
arrS=[]
for i in range(0,7,1):
    if(i==1):
        continue
    arrP.append(arr("_"+str(i)+"p.txt"))
    arrS.append(arr("_"+str(i)+"s.txt"))

#print(arrP[0])
dataP = {
    
    'AOD_Net[6]':arrP[2],
    'FFA_Net[15]':arrP[3],
    'PFF_Net[16]':arrP[4],
    'GMAN[14]':arrP[5],
    'CLAHE_RGB_AOD_Net':arrP[0],
    #'RGB_CLAHE_Small':arrP[1],
    'CLAHE_YCbCr_AODNet':arrP[1]
    }

dataS = {
    'AOD_Net[6]':arrP[2],
    'FFA_Net[15]':arrP[3],
    'PFF_Net[16]':arrP[4],
    'GMAN[14]':arrP[5],
    'CLAHE_RGB_AOD_Net':arrP[0],
    #'RGB_CLAHE_Small':arrP[1],
    'CLAHE_YCbCr_AODNet':arrP[1]
    }


X=[]
for i in range(1,149,1):
    X.append(i)
    
avgP=[]
avgS=[]

for i in range(len(arrP)):
    avgP.append(sum(arrP[i])/len(arrP[i]))
    avgS.append(sum(arrS[i])/len(arrS[i]))
    print(avgP[i],"\t",avgS[i])
for i in range(2):
    avgP.append(avgP[0])
    avgP.remove(avgP[0])
    avgS.append(avgS[0])
    avgS.remove(avgS[0])
'''
dataP = {
    'CLAHE_RGB_AOD_Net':avgP[0],
    #'RGB_CLAHE_Small':arrP[1],
    'CLAHE_YCbCr_AODNet':avgP[1],
    'AOD_Net':avgP[2],
    'FFA_Net':avgP[3],
    'PFF_Net':avgP[4],
    'Gman':avgP[5]
    }
'''

dataAVG = {
    'AOD_Net[6]':[avgS[0],avgP[0]],
    'FFA_Net[15]':[avgS[1],avgP[1]],
    'PFF_Net[16]':[avgS[2],avgP[2]],
    'GMAN[14]':[avgS[3],avgP[3]],
    'CLAHE_RGB_AOD_Net':[avgS[4],avgP[4]],
    'CLAHE_YCbCr_AODNet':[avgS[5],avgP[5]]
    }
'''
methods=['AOD_Net[6]','FFA_Net[15]','PFF_Net[16]','GMAN[14]','CLAHE_RGB_AOD_Net','CLAHE_YCbCr_AODNet']
plt.bar(methods, avgS, width = 0.5)
plt.title('Comparsion of average SSIM metric')
plt.xlabel('Methods')
plt.ylabel('Average SSIM values')
plt.ylim(0.7,0.95)
plt.show()
'''
'''
print(len(X))
plt.plot(X, arrS[2], label = "AOD_Net[6]")
plt.plot(X, arrS[3], label = "FFA_Net[15]")
plt.plot(X, arrS[4], label = "PFF_Net[16]")
plt.plot(X, arrS[5], label = "Gman[14]")
plt.plot(X, arrS[0], label = "CLAHE_RGB_AOD_Net")
#plt.plot(X, arrS[1], label = "RGB_CLAHE_Small")
plt.plot(X, arrS[1], label = "CLAHE_YCbCr_AODNet")

plt.xlabel('Average for number of images/35')
# naming the y axis
plt.ylabel('SSIM_Values')
# giving a title to my graph
plt.title('SSIM_Values_Comparison')
 
# show a legend on the plot
plt.legend()
 
# function to show the plot
plt.show()
'''

#table = tabulate(dataP, headers='keys', showindex=True, tablefmt='fancy_grid')
table1 = tabulate(dataAVG, headers='keys', showindex=False, tablefmt='fancy_grid')
#print(table)
print(table1)

'''
DataFrame(dataP).to_excel('PSNR_Comparison.xlsx')
DataFrame(dataS).to_excel('SSIM_Comparison.xlsx')
'''
