import cv2 

img = cv2.imread('cat.jpg')

# (2657 hei, 1771 wei, 3 rgb)
print(img.shape) 
# ピクセル値：b g r [174 163 165]
print(img[0,10])
# 画像を作成する方法


text = 'hello cat'
org = (0,300)
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 10
color = (255,55,255)
thickness =10
lineType = cv2.LINE_8

res = cv2.putText(img, text,org,fontFace,fontScale,color,thickness,lineType)

cv2.imwrite('res.jpg',res)