from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import cv2
import imutils

ANSWER_KEY={0:1,1:4,2:0,3:3,4:1}

image=cv2.imread("/home/shivendra/Documents/vscode/img_repo/omr.jpeg")
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blurred=cv2.GaussianBlur(gray,(5,5),0)
edged=cv2.Canny(blurred,75,200)

cnts=cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts=imutils.grab_contours(cnts)
docCnt=None

if len(cnts)>0:
    cnts=sorted(cnts,key=cv2.contourArea,reverse=True)
    for c in cnts:
        peri=cv2.arcLength(c,True)
        approx=cv2.approxPolyDP(c,0.02*peri,True)

        if len(approx)==4:
            docCnt=approx
            break

"""cv2.drawContours(image, [docCnt],-1,(0,0,255),2)
cv2.imshow("Outline",image)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

paper=four_point_transform(image,docCnt.reshape(4,2))
warped=four_point_transform(gray,docCnt.reshape(4,2))


"""cv2.imshow("Paper",paper)
cv2.imshow("Gray",warped)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

thresh=cv2.threshold(warped,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

"""cv2.imshow("Binarization",thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

#find contours of bubbles to evaluate the exam

cnts=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts=imutils.grab_contours(cnts)
questionCnts=[]

#loop over the detected contours
for c in cnts:
    (x,y,w,h)=cv2.boundingRect(c)
    ar=w/float(h)

    #in order to label a contour as a question region should be sufficiently wide and sufficiently tall
    #aspect ratio of 1
    if w>=20 and h>=20 and ar>0.9 and ar<1.1:
        questionCnts.append(c)


questionCnts=contours.sort_contours(questionCnts,method="top-to-bottom")[0]
correct=0

for (q,i) in enumerate(np.arange(0,len(questionCnts),5)):
    cnts=contours.sort_contours(questionCnts[i:i+5])[0]
    bubbled=None
    
    for (j,c) in enumerate(cnts):
        mask=np.zeros(thresh.shape,dtype="uint8")
        cv2.drawContours(mask,[c],-1,255,-1)
        mask=cv2.bitwise_and(thresh,thresh,mask=mask)
        total=cv2.countNonZero(mask)

        if bubbled is None or total > bubbled[0]:
            bubbled = (total,j)

    color=(0,0,255)
    k=ANSWER_KEY[q]

    if k==bubbled[1]:
        color=(0,255,0)
        correct+=1
            
    cv2.drawContours(paper,[cnts[k]],-1,color,3)


score=(correct/5.0)*100
print("[INFO] Score : {:.2f}%".format(score))
cv2.putText(paper,"{:.2f}".format(score),(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)
#cv2.imshow("Original",image)
cv2.imshow("Exam",paper)
cv2.waitKey(0)
