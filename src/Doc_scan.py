from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils


#Required Functions

def order_points(pts):
    rect=np.zeros((4,2),dtype="float32")

    s=pts.sum(axis=1)
    rect[0]=pts[np.argmin(s)]
    rect[2]=pts[np.argmax(s)]

    diff=np.diff(pts,axis=1)
    rect[1]=pts[np.argmin(diff)]
    rect[3]=pts[np.argmax(diff)]

    return rect

def four_point_transform(image,pts):

    rect=order_points(pts)
    (bl,br,tr,tl)=rect

    widthA=np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst=np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]],dtype="float32")
    #src=np.array([bl,br,tr,tl],dtype="float32")
    
    M=cv2.getPerspectiveTransform(rect,dst)
    warped=cv2.warpPerspective(image,M,(maxWidth,maxHeight))
    
    print(rect)
    print(dst)
    return warped



"""ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Path to image")
args=vars(ap.parse_args())"""

image=cv2.imread("/home/shivendra/Documents/vscode/img_repo/testimg.jpg")
ratio = image.shape[0]/500.0
orig = image.copy()
image=imutils.resize(image,height=500)


gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray=cv2.GaussianBlur(gray,(5,5),0)
edged=cv2.Canny(gray,75,200)

"""print("Step 1:Edge Detection")
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

print("Step2 : Find Contours of paper")
cv2.drawContours(image, [screenCnt],-1,(0,255,0),2)
print(screenCnt)
cv2.imshow("Outline",image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Start Here
warped = four_point_transform(orig,screenCnt.reshape(4,2)*ratio)
warped = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
T=threshold_local(warped,11,offset=10,method="gaussian")
warped = (warped > T).astype("uint8") * 255

#show
print("Step 3: Apply perspective transform")
#cv2.imshow("Original",imutils.resize(orig,height=650))
cv2.imshow("Scanned",imutils.resize(warped,height=650))
cv2.waitKey(0)
