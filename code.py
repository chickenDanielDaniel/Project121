import cv2
import time
import numpy as np 

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter('output.avi',fourcc , 20, (640,480))

#Start the webcam
cap = cv2.VideoCapture(0)

time.sleep(2)
bg = 0

for i in range(60):
    ret, bg = cap.read()
    
bg = np.flip(bg, axis = 1)

while cap.isOpened():
    ret, img = cap.read()
    img = np.flip(img, axis = 1)

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)   

    l_black = np.array([30,30,0])
    u_black = np.array([104,153,70])
    mask_1 = cv2.inRange(hsv, l_black, u_black)

    l_black = np.array([170,100,100])
    ublack = np.array([180,255,255])
    mask_2 = cv2.inRange(hsv, l_black, u_black)

    mask_1 = mask_1+mask_2

    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, np.ones((3,3),int)) 
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_DILATE, np.ones((3,3),int)) 

    mask_2 = cv2.bitwise_not(mask_1)

    res1 = cv2.bitwise_and(img,img,mask = mask_2)

    res2 = cv2.bitwise_and(bg,bg,mask = mask_1)

    final_output = cv2.addWeighted(res1,1,res2,1,0)
    
    output_file.write(final_output)

    cv2.imshow("Project", final_output)

    cv2.waitKey(1)

cap.release()
out.release()
cv2.destroyAllWindows()