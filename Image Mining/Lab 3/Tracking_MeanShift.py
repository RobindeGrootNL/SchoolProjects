import numpy as np
import cv2

import matplotlib.pyplot as plt
from collections import defaultdict

roi_defined = False
 
def define_ROI(event, x, y, flags, param):
	global r,c,w,h,roi_defined
	# if the left mouse button was clicked, 
	# record the starting ROI coordinates 
	if event == cv2.EVENT_LBUTTONDOWN:
		r, c = x, y
		roi_defined = False
	# if the left mouse button was released,
	# record the ROI coordinates and dimensions
	elif event == cv2.EVENT_LBUTTONUP:
		r2, c2 = x, y
		h = abs(r2-r)
		w = abs(c2-c)
		r = min(r,r2)
		c = min(c,c2)  
		roi_defined = True

cap = cv2.VideoCapture('C:/Users/Robin-PC/OneDrive - Universiteit Twente/6. Year 4/1. Trimester 1/3. OPT3 Image mining or indexing/1. OPT3 Labs/3. OPT3 Lab 3/Sequences/VOT-Ball.mp4')

# take first frame of the video
ret,frame = cap.read()
# load the image, clone it, and setup the mouse callback function
clone = frame.copy()
cv2.namedWindow("First image")
cv2.setMouseCallback("First image", define_ROI)
 
# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("First image", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the ROI is defined, draw it!
	if (roi_defined):
		# draw a green rectangle around the region of interest
		cv2.rectangle(frame, (r, c), (r+h, c+w), (0, 255, 0), 2)
	# else reset the image...
	else:
		frame = clone.copy()
	# if the 'q' key is pressed, break from the loop
	if key == ord("q"):
		break
 
track_window = (r,c,h,w)
# set up the ROI for tracking
roi = frame[c:c+h, r:r+w]
# conversion to Hue-Saturation-Value space
# 0 < H < 180 ; 0 < S < 255 ; 0 < V < 255
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# computation mask of the histogram:
# Pixels with S<30, V<20 or V>235 are ignored 
mask = cv2.inRange(hsv_roi, np.array((0.,10.,5.)), np.array((180.,255.,250.)))
# Marginal histogram of the Hue component
roi_hist = cv2.calcHist([hsv_roi],[2],mask,[180],[0,180])
# Histogram values are normalised to [0,255]
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria: either 10 iterations,
# or move by less than 1 pixel
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1, 1 )

############
# Question 4
############

#### Constructing the initial R-Table of the ROI
roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
ref_point = np.array((roi_gray.shape[0]/2, roi_gray.shape[1]/2))

# Calculate gradient
kernel_y = np.array([[-1], [0], [1]])
kernel_x = kernel_y.T

grad_x_roi = cv2.filter2D(roi_gray, cv2.COLOR_BGR2GRAY, kernel_x)
grad_y_roi = cv2.filter2D(roi_gray, cv2.COLOR_BGR2GRAY, kernel_y)
gradient_angle_roi = (np.arctan2(grad_y_roi, grad_x_roi)+np.pi)/(2*np.pi)
gradient_magnitude_roi = cv2.magnitude(grad_x_roi, grad_y_roi)

# Initialize quantization
N=180
quant = np.linspace(gradient_angle_roi.min(), gradient_angle_roi.max(), num=N)

gradient_angle_roi_quant = np.digitize(gradient_angle_roi, quant)

# Construct r-table
threshold_hough = 80.0

r_table = defaultdict(list)
for i in range(gradient_magnitude_roi.shape[0]):
    for j in range(gradient_magnitude_roi.shape[1]):
        if gradient_magnitude_roi[i, j] > threshold_hough:
            r_table[gradient_angle_roi_quant[i, j]].append((ref_point[0] - i, 
                   ref_point[1] - j))

cpt = 1
while(1):
    ret ,frame = cap.read()
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        #cv2.imshow("Hue image", hsv)
	# Backproject the model histogram roi_hist onto the 
	# current image hsv, i.e. dst(x,y) = roi_hist(hsv(0,x,y))
        dst = cv2.calcBackProject([hsv],[1],roi_hist,[0,180],1)
        
        #cv2.imshow("Weights", dst)
                
        bw_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        kernel_y = np.array([[-1], [0], [1]])
        kernel_x = kernel_y.T
        
        #cv2.imshow('Frame before filtering', bw_frame)
                
        ############
        # Question 3
        ############
		# calculationg the gradient angle and gradient magnitude
        grad_x = cv2.filter2D(bw_frame, cv2.COLOR_BGR2GRAY, kernel_x)
        grad_y = cv2.filter2D(bw_frame, cv2.COLOR_BGR2GRAY, kernel_y)
        
        gradient_angle = (np.arctan2(grad_y, grad_x)+np.pi)/(2*np.pi)
        
        gradient_magnitude = cv2.magnitude(grad_y, grad_x)
                                
        #cv2.imshow("gradient angle", gradient_angle / gradient_angle.max())
        
        #cv2.imshow("gradient magnitude", 
                   #gradient_magnitude / gradient_magnitude.max())
                
		# Make the mask
        threshold = 50
		
        _, thresh = cv2.threshold(gradient_magnitude, 
                                 threshold, 
                                 255, 
                                 cv2.THRESH_BINARY)
        
        gradient_angle_RGB = cv2.merge((gradient_angle,
                                    gradient_angle,
                                    gradient_angle))
                        
        # Create a masked array of the gradient angle, and the masked values
        # are converted to the BGR code of red
        frame_map = gradient_angle_RGB.copy()
        
        frame_map[(thresh == 0)] = np.array([0, 0, 1])
        
        # cv2.imshow('Mask', frame_map)
        
        ############
        # Question 4
        ############
        # Update the r-table
        gradient_angle_quant = np.digitize(gradient_angle, quant)

        H = np.zeros((bw_frame.shape))
        for (i, j), _ in np.ndenumerate(bw_frame):
            magnitude = gradient_magnitude[i, j]
            if magnitude > threshold_hough:
                angle = gradient_angle_quant[i, j]
                for vector in r_table[angle]:
                    V_i, V_j = i + vector[0], j + vector[1]
                    if V_i < H.shape[0] and V_j < H.shape[1]:
                        H[int(V_i), int(V_j)] += 1
        
        track_hough = np.unravel_index(np.argmax(H), H.shape)
        
        cv2.imshow("Hough", H)                
        # apply meanshift to dst to get the new location
        ret, track_window = cv2.meanShift(H, track_window, term_crit)
        
        # Draw a blue rectangle on the current image
        x,y,h,w = track_window
        frame_tracked = cv2.rectangle(frame, (x,y), (x+h,y+w), (255,0,0) ,2)
        cv2.imshow('Sequence',frame_tracked)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('Frame_%04d.png'%cpt,frame_tracked)
        cpt += 1
    else:
        break

cv2.destroyAllWindows()
cap.release()