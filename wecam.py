import cv2 

cap = cv2.VideoCapture(0)
# print(cap.isOpened())

while cap.isOpened():
    success, image = cap.read()
    
    if not success:
        print("うまく画像が読まれてない")
        break
    
    image = cv2.resize(image, dsize=None, fx=0.5 ,fy =0.5)

    image = cv2.flip(image,1)
    cv2.imshow('play show', image)

    if cv2.waitKey(5) & 0xFF ==27:
        break


cap.release()
cv2.destroyAllWindows()


