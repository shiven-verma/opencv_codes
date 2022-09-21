from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils


"""ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Path to image")
args=vars(ap.parse_args())"""

image=cv2.imread("/home/shivendra/Documents/vscode/img_repo/testimg.jpg")
ratio =image.shape[0]/500.0
orig = image.copy()
image=imutils.resize(image,height=500)


gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray=cv2.GaussianBlur(gray,(5,5),0)
edged=cv2.Canny(gray,75,200)

""""print("Step 1:Edge Detection")
cv2.imshow("testimg",image)
cv2.imshow("edged",edged)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

cnts=cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnts=imutils.grab_contours(cnts)
cnts=sorted(cnts,key=cv2.contourArea, reverse=True)[:5]
for c in cnts:
    peri=cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,0.02*peri,True)
    if len(approx) == 4:
        screenCnt=approx
        break

"""print("Step2 : Find Contours of paper")
cv2.drawContours(image, [screenCnt],-1,(0,255,0),2)
cv2.imshow("Outline",image)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

screenCnt.reshape(4,2)*ratio
