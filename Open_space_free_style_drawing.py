import numpy as np
import cv2
from collections import deque

#default called trackbar function 
def setValues(x):
   print("")


# Creating the trackbars needed for adjusting the marker colour
cv2.namedWindow("Color detectors")
cv2.createTrackbar("Upper Hue", "Color detectors", 153, 180,setValues)
cv2.createTrackbar("Upper Saturation", "Color detectors", 255, 255,setValues)
cv2.createTrackbar("Upper Value", "Color detectors", 255, 255,setValues)
cv2.createTrackbar("Lower Hue", "Color detectors", 64, 180,setValues)
cv2.createTrackbar("Lower Saturation", "Color detectors", 72, 255,setValues)
cv2.createTrackbar("Lower Value", "Color detectors", 49, 255,setValues)


# Giving different arrays to handle colour points of different colour
bpts = [deque(maxlen=1024)]
gpts = [deque(maxlen=1024)]
rpts = [deque(maxlen=1024)]
ypts = [deque(maxlen=1024)]

# These indexes will be used to mark the points in particular arrays of specific colour
blue_idx = 0
green_idx = 0
red_idx = 0
yellow_idx = 0

#The kernel to be used for dilation purpose 
kernel = np.ones((5,5) ,np.uint8)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIdx = 0

# Setup for canvas
paintWind = np.zeros((471, 636, 3)) + 255
paintWind = cv2.rectangle(paintWind, (40, 1), (140, 65), (0, 0, 0), 2)
paintWind = cv2.rectangle(paintWind, (160, 1), (255, 65), colors[0], -1)
paintWind = cv2.rectangle(paintWind, (275, 1), (370, 65), colors[1], -1)
paintWind = cv2.rectangle(paintWind, (390, 1), (485, 65), colors[2], -1)
paintWind = cv2.rectangle(paintWind, (505, 1), (600, 65), colors[3], -1)

cv2.putText(paintWind, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWind, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWind, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWind, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWind, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2, cv2.LINE_AA)
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)


# Load default webcamera
cap = cv2.VideoCapture(0)

# run loop..
while True:
    # Read frame from  front camera.
    ret, frame = cap.read()
    #Flipping frame to see as side of us
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    u_hue = cv2.getTrackbarPos("Upper Hue", "Color detectors")
    u_saturation = cv2.getTrackbarPos("Upper Saturation", "Color detectors")
    u_value = cv2.getTrackbarPos("Upper Value", "Color detectors")
    l_hue = cv2.getTrackbarPos("Lower Hue", "Color detectors")
    l_saturation = cv2.getTrackbarPos("Lower Saturation", "Color detectors")
    l_value = cv2.getTrackbarPos("Lower Value", "Color detectors")
    Upper_hsv = np.array([u_hue,u_saturation,u_value])
    Lower_hsv = np.array([l_hue,l_saturation,l_value])


    # Add color buttons in window display
    frame = cv2.rectangle(frame, (40,1), (140,65), (122,122,122), -1)
    frame = cv2.rectangle(frame, (160, 1), (255, 65), colors[0], -1)
    frame = cv2.rectangle(frame, (275,1), (370,65), colors[1], -1)
    frame = cv2.rectangle(frame, (390,1), (485,65), colors[2], -1)
    frame = cv2.rectangle(frame, (505,1), (600,65), colors[3], -1)
    cv2.putText(frame, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2, cv2.LINE_AA)


    # Identify pointer byy making its mask.
    Msk = cv2.inRange(hsv, Lower_hsv, Upper_hsv)
    Msk = cv2.erode(Msk, kernel, iterations=1)
    Msk = cv2.morphologyEx(Msk, cv2.MORPH_OPEN, kernel)
    Msk = cv2.dilate(Msk, kernel, iterations=1)

    # Find contours for pointr after idetify it
    cnts,_ = cv2.findContours(Msk.copy(), cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)
    center = None

    # If the contours are formed
    if len(cnts) > 0:
    	# sorting contours to find largest
        cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        # Get the radius of enclosing circle around found contours
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        # Draw circle around contour
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        # Calculating center of detected contours
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        # Now checking if user wants to click on any button top of the screen
        if center[1] <= 65:
            if 40 <= center[0] <= 140: # Clear Button
                bpts = [deque(maxlen=512)]
                gpts = [deque(maxlen=512)]
                rpts = [deque(maxlen=512)]
                ypts = [deque(maxlen=512)]

                blue_idx = 0
                green_idx = 0
                red_idx = 0
                yellow_idx = 0

                paintWind[67:, :, :] = 255
            elif 160 <= center[0] <= 255:
                    colorIdx = 0 # Blue
            elif 275 <= center[0] <= 370:
                    colorIdx = 1 # Green
            elif 390 <= center[0] <= 485:
                    colorIdx = 2 # Red
            elif 505 <= center[0] <= 600:
                    colorIdx = 3 # Yellow
        else :
            if colorIdx == 0:
             bpts[blue_idx].appendleft(center)
            elif colorIdx == 1:
                gpts[green_idx].appendleft(center)
            elif colorIdx == 2:
                rpts[red_idx].appendleft(center)
            elif colorIdx == 3:
                ypts[yellow_idx].appendleft(center)
    # Append the next deques when nothing is detected to avois messing up
    else:
        bpts.append(deque(maxlen=512))
        blue_idx += 1
        gpts.append(deque(maxlen=512))
        green_idx += 1
        rpts.append(deque(maxlen=512))
        red_idx += 1
        ypts.append(deque(maxlen=512))
        yellow_idx += 1

    # Draw lines of all colors on the canvas and also in frame..
    points = [bpts, gpts, rpts, ypts]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWind, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    # It will Show all windows
    cv2.imshow("Tracking", frame)
    cv2.imshow("Paint", paintWind)
    cv2.imshow("mask", Msk)

	# If the 'q' key is pressed then stop the application 
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Drop camera and all resources.
cap.release()
cv2.destroyAllWindows()