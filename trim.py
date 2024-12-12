import cv2 

img = cv2.imread('cat.jpg')

print(img.shape)

res = img[1:950, 10:1200]

cv2.imwrite('test.jpg',res)