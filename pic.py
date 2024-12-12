import cv2 
import numpy as np

img = cv2.imread('cat.jpg')

noise = img.copy()
height = img.shape[0]
width = img.shape[1]


white =(255,255,255)
black = (0,0,0)


nx  = np.random.randint(0,width-1,00)
ny  = np.random.randint(0,height-1,1900)

noise[(ny,nx)] =white

nx  = np.random.randint(0,width-1,1500)
ny  = np.random.randint(0,height-1,1500)
noise[(ny,nx)] =black




cv2.imwrite('noise.jpg',noise)