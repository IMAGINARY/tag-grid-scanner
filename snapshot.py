import cv2

capture = cv2.VideoCapture(0)


while(True):
    ret, frame = capture.read()
    
    cv2.imshow('video', frame)
    
    key = cv2.waitKey(1)
    if key == 32 or key == 13:
        cv2.imwrite('snapshot.jpg', frame)
        key = cv2.waitKey(1000)
        
    if key == 27:
        break;

capture.release()
cv2.destroyAllWindows()
