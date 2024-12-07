# Code adapted from: https://github.com/bitcraze/crazyflie-lib-python/blob/master/examples/autonomousSequence.py

import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

# CrazyFlie imports:

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.positioning.position_hl_commander import PositionHlCommander

group_number = 22
min_y_pos = -0.5
max_y_pos = 0.1

# Possibly try 0, 1, 2 ...
camera = 0

cap = cv2.VideoCapture(camera)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    middle = frame[:, 165:315] # only looks at middle 1/3ish of x axis; takes entire y-axis
    
    # These define the upper and lower HSV for the red obstacles.
    # Note that the red color wraps around 180, so there are two intervals.
    # Tuning of these values will vary depending on the camera.
    lb1 = (145, 35, 75)
    ub1 = (180, 255, 255)
    lb2 = (0, 75, 75)
    ub2 = (20, 255, 255)

    # Perform contour detection on the input frame.
    #hsv1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #hsv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv1 = cv2.cvtColor(middle, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(middle, cv2.COLOR_BGR2HSV)

    # Compute mask of red obstacles in either color range.
    mask1 = cv2.inRange(hsv1, lb1, ub1)
    mask2 = cv2.inRange(hsv2, lb2, ub2)
    # Combine the masks.
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Compute
    cv2.imshow('mask', mask)    

    # Hit q to quit.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()

import sys

# Get the current crazyflie position:
def position_estimate(scf):
    log_config = LogConfig(name='Kalman Variance', period_in_ms=500)
    log_config.add_variable('kalman.varPX', 'float')
    log_config.add_variable('kalman.varPY', 'float')
    log_config.add_variable('kalman.varPZ', 'float')

    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]
            x = data['kalman.varPX']
            y = data['kalman.varPY']
            z = data['kalman.varPZ']
            
    print(x, y, z)
    return x, y, z


# Set the built-in PID controller:
def set_PID_controller(cf):
    # Set the PID Controller:
    print('Initializing PID Controller')
    cf.param.set_value('stabilizer.controller', '1')
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(2)
    return


# Ascend and hover at 1m:
def ascend_and_hover(cf):
    # Ascend:
    for y in range(5):
        cf.commander.send_hover_setpoint(0, 0, 0, y / 10)
        time.sleep(0.1)
    # Hover at 0.5 meters:
    for _ in range(20):
        cf.commander.send_hover_setpoint(0, 0, 0, 0.5)
        time.sleep(0.1)
    return


# Sort through contours in the image
def findGreatesContour(contours):
    largest_area = 0
    largest_contour_index = -1
    i = 0
    total_contours = len(contours)

    while i < total_contours:
        area = cv2.contourArea(contours[i])
        if area > largest_area:
            largest_area = area
            largest_contour_index = i
        i += 1

    #print(largest_area)

    return largest_area, largest_contour_index


# Find contours in the image
def check_contours(frame):

    print('Checking image:')

    # These define the upper and lower HSV for the red obstacles.
    # Note that the red color wraps around 180, so there are two intervals.
    # Tuning of these values will vary depending on the camera.
    lb1 = (145, 35, 75)
    ub1 = (180, 255, 255)
    lb2 = (0, 75, 75)
    ub2 = (20, 255, 255)

    # Perform contour detection on the input frame.
    hsv1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Compute mask of red obstacles in either color range.
    mask1 = cv2.inRange(hsv1, lb1, ub1)
    mask2 = cv2.inRange(hsv2, lb2, ub2)
    # Combine the masks.
    mask = cv2.bitwise_or(mask1, mask2)

    # Use the OpenCV findContours function.
    # Note that there are three outputs, but we discard the first one.    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    largest_area, largest_contour_index = findGreatesContour(contours)
    contour_x = -1
    
    if largest_contour_index != -1:
        contour_x = np.mean(contours[largest_contour_index], axis=0).flatten()[0]
    
        #print(largest_area)
    
        if largest_area > 100:
            return mask, True, contour_x
            
    return mask, False, contour_x


# Follow the setpoint sequence trajectory:
def adjust_position(cf, current_x, current_y, direction):

    print('Adjusting position')

    steps_per_meter = int(10)
    # Set the number here (the iterations of the for-loop) to the number of side steps.
    # You may choose to tune the number and size of the steps.
    for i in range(3): 
        if direction == "RIGHT":
            current_y = current_y - 1.0/float(steps_per_meter)   
        elif direction == "FORWARD":
            current_x = current_x + 1.0/float(steps_per_meter)
        elif direction == "LEFT":
            current_y = current_y + 1.0/float(steps_per_meter)

        position = [current_x, current_y, 0.5, 0.0]
                        
        print('Setting position {}'.format(position))
        for i in range(10):
            """
            cf.commander.send_position_setpoint(position[0],
                                                position[1],
                                                position[2],
                                                position[3])
            """
            time.sleep(0.1)

    #cf.commander.send_stop_setpoint()
    # Make sure that the last packet leaves before the link is closed.
    # The message queue is not flushed before closing.
    #time.sleep(0.05)
    return current_x, current_y


# Hover, descend, and stop all motion:
def hover_and_descend(cf):
    print('Descending:')
    # Hover at 0.5 meters:
    for _ in range(30):
        cf.commander.send_hover_setpoint(0, 0, 0, 0.5)
        time.sleep(0.1)
    # Descend:
    for y in range(10):
        cf.commander.send_hover_setpoint(0, 0, 0, (10 - y) / 25)
        time.sleep(0.1)
    # Stop all motion:
    for i in range(10):
        cf.commander.send_stop_setpoint()
        time.sleep(0.1)
    return

""" going off of lab 8 UP TO DATE VERSION """
import cv2
import time
import numpy as np

"""
# load the COCO class names
with open('Lab8_Supplement/object_detection_classes_coco.txt', 'r') as f:
    class_names = f.read().split('\n')

# get a different color array for each of the classes
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

# load the DNN model
#model = cv2.dnn.readNet(model='Lab8_Supplement/frozen_inference_graph.pb',
#                        config='Lab8_Supplement/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', 
#  framework='TensorFlow')                     
"""
# ************ Parameters that might be useful to change ************ 
# COCO label id that we want to track
tracking_label = 1 # PERSON (1), CHAIR (62)

# Set the URI the Crazyflie will connect to
group_number = 21
uri = f'radio://0/21/2M'

# Possibly try 0, 1, 2 ...
camera = 0

# Confidence of detection
confidence = 0.7

# ******************************************************************

# Initialize all the CrazyFlie drivers:
cflib.crtp.init_drivers(enable_debug_driver=False)

# Scan for Crazyflies in range of the antenna:
print('Scanning interfaces for Crazyflies...')
available = cflib.crtp.scan_interfaces()

# List local CrazyFlie devices:
print('Crazyflies found:')
for i in available:
    print(i[0])

if len(available) == 0:
    print('No Crazyflies found, cannot run example')
else:
    ## Ascend to hover; run the sequence; then descend from hover:
    # Use the CrazyFlie corresponding to team number:
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        # Get the Crazyflie class instance:
        cf = scf.cf

        # Initialize and ascend:
        t = time.time()
        elapsed = time.time() - t
        ascended_bool = 0
        direction = "RIGHT"

        # capture the video
        cap = cv2.VideoCapture(camera)
        
        # get the video frames' width and height
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        # flag indicating whether to exit the main loop and then descend
        exit_loop = False

        # Ascend and hover a bit
        #set_PID_controller(cf)
        #ascend_and_hover(cf)
        #time.sleep(1)
        
        current_x = 0
        current_y = 0
        
        # detect objects in each frame of the video
        while cap.isOpened() and not exit_loop:
            
            # Try to read image
            ret, frame = cap.read()
            if ret:

                mask, contour_bool, contour_x = check_contours(frame)
                # if drone is near boundaries, move towards center
                if current_y < min_y_pos:
                    direction = "LEFT"
                    current_x, current_y = adjust_position(cf, current_x, current_y, direction)
                    current_x, current_y = adjust_position(cf, current_x, current_y, direction)
                    
                elif current_y > max_y_pos:
                    direction = "RIGHT"
                    current_x, current_y = adjust_position(cf, current_x, current_y, direction)
                    current_x, current_y = adjust_position(cf, current_x, current_y, direction)

                else:  
                    if(contour_bool):
                        print("theres a contour")
                        # if obstacle is not centered in image, move forward
                        if contour_x < 210 or contour_x > 450:
                            direction = "FORWARD"
                        elif contour_x < 330:
                            direction = "RIGHT"
                        else:

                    # if no obstacle is visible, move forward
                    else:
                        direction = "FORWARD"

                    print("going %s" %(direction))
                    current_x, current_y = adjust_position(cf, current_x, current_y, direction)

                # Check image
                cv2.imshow('mask', mask)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
            else:
                print('no image!!')
                
        cap.release()
        
        # Descend and stop all motion:
        print('target reached!')
        #hover_and_descend(cf)
        
        cv2.destroyAllWindows()