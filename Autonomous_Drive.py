import cv2, time, math
import numpy as np
from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar import QLabsQCar
from qvl.free_camera import QLabsFreeCamera
from qvl.basic_shape import QLabsBasicShape
from qvl.system import QLabsSystem
from qvl.walls import QLabsWalls
from qvl.flooring import QLabsFlooring
from qvl.stop_sign import QLabsStopSign
from qvl.crosswalk import QLabsCrosswalk
from qvl.traffic_cone import QLabsTrafficCone
from ultralytics import YOLO

from Setup_Competition import *
import subprocess
import os

def detection(car, model):
    _, img = car.get_image(4)
    result = model.predict(source=img, save=False)
    try:
        obj_list = ['red', 'green', 'stopsign', 'stoplane', 'crosswalk', 'cone']
        box_color_list = [(50,50,255), (0,204,0), (194,153,255), (255,204,51), (255,102,204), (0,153,255)]
        det_result_obj = []
        det_result_size = []
        det_result_coord = result[0].boxes.xyxy.tolist()

        for i in range(len(result[0].boxes.cls.tolist())):
            x1 = int(result[0].boxes.xyxy.tolist()[i][0])
            x2 = int(result[0].boxes.xyxy.tolist()[i][2])
            y1 = int(result[0].boxes.xyxy.tolist()[i][1])
            y2 = int(result[0].boxes.xyxy.tolist()[i][3])
            det_result_obj.append(int(result[0].boxes.cls[i]))
            det_result_size.append(round((x2-x1)*(y2-y1)/(img.shape[0]*img.shape[1])*100,3))

            box_color = box_color_list[int(result[0].boxes.cls[i])]
            text_color = (0, 0, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
            txt_loc = (max(x1+2, 0), max(y1+2, 0))
            txt = obj_list[int(result[0].boxes.cls[i])]
            img_h, img_w, _ = img.shape
            if txt_loc[0] >= img_w or txt_loc[1] >= img_h:
                # cv2.imshow('result', img)
                cv2.waitKey(1)
                return [det_result_obj, det_result_size, det_result_coord]
            margin = 3
            size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
            w = size[0][0] + margin * 2
            h = size[0][1] + margin * 2
            cv2.rectangle(img, (x1-1, y1-1-h), (x1+w, y1), box_color, -1)
            cv2.putText(img, txt, (x1+margin, y1-margin-2), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, lineType=cv2.LINE_AA)
        # cv2.imshow('result', img)
        # cv2.waitKey(1)
        return [det_result_obj, det_result_size, det_result_coord]
    except Exception:
        return [0]

def find_nearest_point_index(path_points, current_position):
    distances = np.linalg.norm(path_points - current_position, axis=1)
    return np.argmin(distances)

def find_target_point(path_points, current_position, ld):
    nearest_point_index = find_nearest_point_index(path_points, current_position)
    for i in range(nearest_point_index, len(path_points)):
        distance = np.linalg.norm(path_points[i] - current_position)
        if distance > ld:
            return (path_points[i], distance)
    return (path_points[2], distance)

def calculate_steering_angle(current_position, target_point, yaw, L, ld, speed):
    ### PURE PURSUIT ###
    target_vector = np.array([target_point[0] - current_position[0], target_point[1] - current_position[1]])
    rotated_x = math.cos(yaw)*target_vector[0] + math.sin(yaw) * target_vector[1]
    rotated_y = math.sin(yaw)*target_vector[0] - math.cos(yaw) * target_vector[1]

    target_angle = np.arctan2(rotated_y, rotated_x)

    delta = np.arctan2((2 * L * np.sin(target_angle)), ld * speed * 1.65) 

    str_max = 0.4

    return max(min(delta, str_max), -str_max), target_angle, rotated_x, rotated_y

def adjust_speed_based_on_steering_angle(steering_angle, current_speed):
    # adjust speed for better lap time
    abs_angle = np.abs(steering_angle)

    angle_threshold = 0.02
    max_angle = 0.35

    max_speed = 1.4
    min_speed = 0.85

    if abs_angle > angle_threshold:
        target_speed = min_speed
    else:
        target_speed = max_speed

    # limit accel. to 0.2 maximum
    max_change = 0.2
    if target_speed > current_speed:
        new_speed = min(target_speed, current_speed + max_change)
    else:
        new_speed = target_speed

    return new_speed

def update_car_state(car, steering_angle, speed, avoidance):
    speed=adjust_speed_based_on_steering_angle(steering_angle,speed)
    if avoidance == True:
        steering_angle -= 0.11
    status, veh_posi, orien, _, _ = car.set_velocity_and_request_state(forward=speed, turn=steering_angle, headlights=False, leftTurnSignal=False, rightTurnSignal=True, brakeSignal=False, reverseSignal=False)
    yaw = orien[2]
    current_position = np.array([veh_posi[0], veh_posi[1]])
    return current_position, yaw, speed

### Object Avoidance ###
def get_front_lidar(car):
    obstacle_ahead = False
    success, angle, distance = car.get_lidar(samplePoints=400)

    angle_deg = 180*angle/np.pi
    angle_deg = np.where(angle_deg > 180, angle_deg - 360, angle_deg)

    angle_front, dist_front = [], []

    i = 0
    for i in range(len(angle_deg)):
        if 0 <= angle_deg[i] <= 30:
            angle_front.append(angle_deg[i])
            dist_front.append(distance[i])

    dist_slice = dist_front[::3]
    angle_slice = angle_front[::3]

    gradient = abs(np.gradient(dist_slice,angle_slice))
    gradient_dif = abs(np.diff(gradient))
    i = 0

    thres = 0.9
    for i in dist_front:
        if i < thres and min(dist_front) >= 0.1:
            if max(gradient) > 0.5*i and max(gradient_dif) > 0.1 : 
                obstacle_ahead = True

    return success, angle_front, dist_front, obstacle_ahead


# Environment Setting
qlabs = QuanserInteractiveLabs()
print("Connecting to QLabs...")
try:
    qlabs.open("localhost")
    print("Connected to QLabs")
except:
    print("Unable to connect to QLabs")
    quit()

car = setup()

x_offset = 0.13
y_offset = 1.67

#============================= traffic cone =============================#
"""
Uncomment this part if you want to validate the obstacle aviodance algorithmn
"""
# TrafficCone = QLabsTrafficCone(qlabs)
# TrafficCone.spawn_degrees([2.1 + x_offset, y_offset-0.5, 0], [0, 0, 0], scale=[.5, .5, .5], configuration=0, waitForConfirmation=True)
#============================= traffic cone =============================#

process = subprocess.Popen(['python', 'Traffic_Lights_Competition.py'])

model = YOLO("yolo.pt")
#cv2.namedWindow('result', flags=cv2.WINDOW_NORMAL)
#cv2.resizeWindow('result', 800, 600)
#cv2.moveWindow('result', 1120, 480)

with open('path.txt', 'r') as file:
    lines = file.readlines()
    path_points = [list(map(float, line.strip().split())) for line in lines]
path_points = np.array(path_points)

L = 0.27     # Wheel Base (m)
ld = 0.65     # Look Ahead Distance for Pure Pursuit (m)
speed = 0.8  # Default Speed

### Initialize ###
car.set_transform_and_request_state_degrees(location=[-1.335 + x_offset, -2.5 + y_offset, 0.005], rotation=[0, 0, -45], enableDynamics=True, headlights=False, leftTurnSignal=False, rightTurnSignal=False, brakeSignal=False, reverseSignal=False, waitForConfirmation=True)
steering_angle = 0
_ = detection(car, model)

current_position, yaw, speed = update_car_state(car, 0, speed, False)
target_point, distance = find_target_point(path_points, current_position, ld)

iteration = 0
avoidance = False
avoidance_prev = []
det_stoplane = 0
det_trafficcone_time = 0

endpoint = np.array((-1.9501, 0.20551))

try:
    ld = 0.3
    speed=0.8
    for _ in range(5):
        start_time1 = time.time()
        target_point, distance = find_target_point(path_points, current_position, ld)
        steering_angle, alpha, rotated_x, rotated_y = calculate_steering_angle(current_position, target_point, yaw, L, ld, speed)
        status, veh_posi, orien, _, _ = car.set_velocity_and_request_state(forward=speed, turn=steering_angle, headlights=False, leftTurnSignal=False, rightTurnSignal=True, brakeSignal=False, reverseSignal=False)
        yaw = orien[2]
        current_position = np.array([veh_posi[0], veh_posi[1]])
        time.sleep(0.01)
        end_time1 = time.time()
        
    lap_start = time.time()
    ld = 0.65
    speed=0.8
    
    while True:
        start_time = time.time()
        steering_angle, alpha, rotated_x, rotated_y = calculate_steering_angle(current_position, target_point, yaw, L, ld, speed)
        det_result = detection(car, model)
        success, angle_front, dist_front, avoidance_result = get_front_lidar(car)

        # Obstacle Aviodance in effect
        avoidance_prev.append(avoidance_result)
        if len(avoidance_prev) >= 3:
            if True in avoidance_prev[-3:]:
                avoidance = True
            elif not any(avoidance_prev[-3:]):
                avoidance_prev = []
                avoidance = avoidance_result
            else:
                avoidance = avoidance_result
        else:
            avoidance = avoidance_result

        ### Motion Planning depending on Obstacle Detection ###
        ## Obstacle Ahead ##
        if len(det_result[0]) > 0:
                if 5 in det_result[0]:
                    det_trafficcone_time = time.time()

                if avoidance and time.time()-det_trafficcone_time >= 3:
                    avoidance = False

                # Traffic Light & Stop Sign detected - Proceed for its abnormal situation
                if ((0 in det_result[0]) or (1 in det_result[0])) and (2 in det_result[0]):
                    target_point, distance = find_target_point(path_points, current_position, ld)
                    steering_angle, alpha, rotated_x, rotated_y = calculate_steering_angle(current_position, target_point, yaw, L, ld, speed)
                    current_position, yaw, speed = update_car_state(car, steering_angle, speed, avoidance)

                # Stop Sign + @ (No Traffic Light)
                elif 2 in det_result[0]:
                    if det_result[1][det_result[0].index(2)] >= 1.3:  
                        car.set_velocity_and_request_state(forward=0, turn=steering_angle, headlights=False, leftTurnSignal=False, rightTurnSignal=True, brakeSignal=False, reverseSignal=False)
                        time.sleep(1)
                        for _ in range(10):
                            target_point, distance = find_target_point(path_points, current_position, ld)
                            steering_angle, alpha, rotated_x, rotated_y = calculate_steering_angle(current_position, target_point, yaw, L, ld, speed)
                            current_position, yaw, speed = update_car_state(car, steering_angle, speed, avoidance)
                            det_result = detection(car, model) 
                    # Proceed
                    else:
                        target_point, distance = find_target_point(path_points, current_position, ld)
                        steering_angle, alpha, rotated_x, rotated_y = calculate_steering_angle(current_position, target_point, yaw, L, ld, speed)
                        current_position, yaw, speed = update_car_state(car, steering_angle, speed, avoidance)

                # Traffic Light + @ (No Stop Sign)
                elif (0 in det_result[0]) or (1 in det_result[0]):
                    # Stop Line detected
                    if 3 in det_result[0]:
                        det_stoplane = 1
                        # Proceed
                        target_point, distance = find_target_point(path_points, current_position, ld)
                        steering_angle, alpha, rotated_x, rotated_y = calculate_steering_angle(current_position, target_point, yaw, L, ld, speed)
                        current_position, yaw, speed = update_car_state(car, steering_angle, speed, avoidance)
                        speed -= 0.375
                    # Stop Line not detected
                    else:
                        # Red Light
                        if 0 in det_result[0]:
                            if det_stoplane == 1:
                                car.set_velocity_and_request_state(forward=0, turn=0, headlights=False, leftTurnSignal=False, rightTurnSignal=True, brakeSignal=False, reverseSignal=False)
                            else:
                                det_stoplane = 0
                                target_point, distance = find_target_point(path_points, current_position, ld)
                                steering_angle, alpha, rotated_x, rotated_y = calculate_steering_angle(current_position, target_point, yaw, L, ld, speed)
                                current_position, yaw, speed = update_car_state(car, steering_angle, speed, avoidance)
                        # Blue Light
                        elif 1 in det_result[0]:
                            if det_stoplane != 0:
                                det_stoplane = 0
                            # Proceed
                            target_point, distance = find_target_point(path_points, current_position, ld)
                            steering_angle, alpha, rotated_x, rotated_y = calculate_steering_angle(current_position, target_point, yaw, L, ld, speed)
                            current_position, yaw, speed = update_car_state(car, steering_angle, speed, avoidance)

                # Traffic Light & Stop Sign both no detected
                # Proceed
                else:
                    target_point, distance = find_target_point(path_points, current_position, ld)
                    steering_angle, alpha, rotated_x, rotated_y = calculate_steering_angle(current_position, target_point, yaw, L, ld, speed)
                    current_position, yaw, speed = update_car_state(car, steering_angle, speed, avoidance)

        ## No Obstacle Ahead ##
        else:
            if avoidance and time.time()-det_trafficcone_time >= 3:
                avoidance = False

            target_point, distance = find_target_point(path_points, current_position, ld)
            steering_angle, alpha, rotated_x, rotated_y = calculate_steering_angle(current_position, target_point, yaw, L, ld, speed)
            current_position, yaw, speed = update_car_state(car, steering_angle, speed, avoidance)
            
        # See if the car has reached to the end
        dist_to_end = np.linalg.norm(current_position - endpoint)
        if dist_to_end < 0.1:
            car.set_velocity_and_request_state(forward=0, turn=0, headlights=False, leftTurnSignal=False, rightTurnSignal=True, brakeSignal=False, reverseSignal=False)
            lap_time = time.time() - lap_start
            print("LAP COMPLETE.\nLAP TIME :", lap_time, "sec")
            break
                
except KeyboardInterrupt:
    car.set_velocity_and_request_state(forward=0, turn=0, headlights=False, leftTurnSignal=False, rightTurnSignal=True, brakeSignal=False, reverseSignal=False)
    #cv2.waitKey(1)
    qlabs.destroy_all_spawned_actors()
except Exception as e:
    car.set_velocity_and_request_state(forward=0, turn=0, headlights=False, leftTurnSignal=False, rightTurnSignal=True, brakeSignal=False, reverseSignal=False)
    #cv2.waitKey(1)
    qlabs.destroy_all_spawned_actors()
    print(e)