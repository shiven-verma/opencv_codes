import cv2
import numpy as np
from collections import OrderedDict

def probables(sorted_cont):
	res = []
	for i in range(len(sorted_cont)-1):
		if not(abs(sorted_cont[i][1][0]-sorted_cont[i+1][1][0])<=2 and abs(sorted_cont[i][1][1]-sorted_cont[i+1][1][1])<=2):
			res.append(sorted_cont[i])
			i -= 1
	probab = []
	scale = cv2.getTrackbarPos('scale_of_det','test') 
	for i in range(len(res)-2):
		if abs(res[i][1][0]-res[i+1][1][0])<=abs(res[i][1][1]-res[i+1][1][1])*scale*0.01 and abs(res[i+1][1][0]-res[i+2][1][0])<=abs(res[i+1][1][1]-res[i+2][1][1])*scale*0.01:
			probab.append((res[i],res[i+1],res[i+2]))

	return probab


cap = cv2.VideoCapture(0)
# var = 0
def onchange(a):
	pass
cv2.namedWindow('test')
cv2.createTrackbar('cont_perimeter', 'test', 40, 1000, onchange)
cv2.createTrackbar('scale_of_det', 'test', 20, 100, onchange)

while(1):
	_, frame = cap.read()
	# It converts the BGR color space of image to HSV color space
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# Threshold of orange in HSV space
	lower_orange = np.array([20, 120, 70])  
	upper_orange = np.array([40, 255, 255])

	# preparing the mask to overlay
	mask = cv2.inRange(hsv, lower_orange, upper_orange)

	# The black region in the mask has the value of 0,
	# so when multiplied with original image removes all non-blue regions
	image_obj = cv2.bitwise_and(frame, frame, mask = mask)
	gray = cv2.cvtColor(image_obj, cv2.COLOR_BGR2GRAY)

	kernel = np.ones((4, 4), np.uint8)
	erosion = cv2.erode(gray, kernel, iterations=2)

	blur = cv2.GaussianBlur(erosion, (5, 5), 0)
	thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

	# Now finding Contours         ###################
	contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	coordinates = []
	centroids = []
	# res_cont = []
	for cnt in contours:
		# [point_x, point_y, width, height] = cv2.boundingRect(cnt)
		approx = cv2.approxPolyDP(
			cnt, 0.07 * cv2.arcLength(cnt, True), True)
		var = cv2.getTrackbarPos('cont_perimeter','test')
		# print(approx)
		if cv2.arcLength(cnt, True)>var: ############################################################# (len(approx) == 4 or len(approx) == 3) and cv2.arcLength(cnt, True)>var
			coordinates.append(cnt)
			cv2.drawContours(image_obj, [cnt], 0, (0, 0, 255), 3)
			M = cv2.moments(cnt)
			if M['m00'] != 0:
				cx = int(M['m10']/M['m00'])
				cy = int(M['m01']/M['m00'])
				# cv.drawContours(image, [i], -1, (0, 255, 0), 2)
				cv2.circle(image_obj, (cx, cy), 4, (0, 0, 255), -1)
				centroids.append((cx,cy))
	sorted_cont = sorted(zip(coordinates,centroids), key=(lambda a: a[1]))
	# print(list(i[1] for i in sorted_cont))
	fixed_cont = probables(sorted_cont)

	for i in fixed_cont:
		cv2.line(image_obj, i[0][1],i[1][1],(0,255,255),2)
		cv2.line(image_obj, i[1][1],i[2][1],(0,255,255),2)

	if fixed_cont:
		image_obj = cv2.putText(image_obj, "${}".format(len(fixed_cont)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255) , 2, cv2.LINE_AA)
	# for i in coordinates:
		
			# cv2.putText(image, "center", (cx - 20, cy - 20),
			#            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
	# print(f"x: {cx} y: {cy}")
	cv2.imshow("result", image_obj)
	cv2.imshow("Original",frame)

	# cv2.imshow('frame', frame)
	# cv2.imshow('mask', mask)
	# cv2.imshow('result', result)
	cv2.waitKey(10)

cv2.destroyAllWindows()
cap.release()