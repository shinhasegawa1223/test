import cv2 

img = cv2.imread('cat.jpg')

pt1=(10,10)
pt2=(150,150)
color = (25,25,255)
thickness=10
lineType = cv2.LINE_8
shift = 0


line_pic = cv2.line(img,pt1,pt2,color,thickness,lineType,shift)
cv2.imwrite('line_pic.jpg',line_pic)