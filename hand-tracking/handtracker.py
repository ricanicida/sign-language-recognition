import cv2
import mediapipe as mp

import numpy as np
import os
from collections import deque
from itertools import islice
from time import time_ns


class HandTracker():
    def __init__(self, video_fps: int=0, hei_sampling_rate: int=0, hei_frame_step: int=1, hei_max_duration: float=0,
                 hei_overlap: float=0, hei_max_frames: int=30, mode: str='IMAGE', max_hands: int=2,
                 detection_con: float=0.5, model_complexity: int=1, track_con: float=0.5) -> None:
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.model_complexity = model_complexity
        self.track_con = track_con
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands,self.model_complexity,
                                        self.detection_con, self.track_con)
        self.results=None

        if hei_max_duration and hei_sampling_rate:
            buffer_length = int(hei_sampling_rate * hei_max_duration)
        else:
            buffer_length = hei_max_frames

        if video_fps and hei_sampling_rate:
            hei_frame_step = int(video_fps/hei_sampling_rate)

        self.hei_frame_step = hei_frame_step
        self.buffer_length = buffer_length
        self.hei_overlap_frame = int((1-hei_overlap)*buffer_length)
        self.hei_frame_count = 0
        self.hei_frame_flag = True
        self.hei_averaging_flag = False
        self.first_hei_flag = True
        
        self.video_frame_count = 0

        self.left_image_buffer = deque(buffer_length*[[]], buffer_length)
        self.right_image_buffer = deque(buffer_length*[[]], buffer_length)

        self.left_center_coordinates = deque(buffer_length*[[]], buffer_length)
        self.right_center_coordinates = deque(buffer_length*[[]], buffer_length)

        self.left_min_box_size_buffer = deque(buffer_length*[-1], buffer_length)
        self.right_min_box_size_buffer = deque(buffer_length*[-1], buffer_length)

        self.left_max_box_size_buffer = deque(buffer_length*[-1], buffer_length)
        self.right_max_box_size_buffer = deque(buffer_length*[-1], buffer_length)


    def hands_finder(self,imageRGB):
        if self.video_frame_count % self.hei_frame_step == 0:
            self.hei_frame_flag = True
            self.results = self.hands.process(imageRGB)
            self.hei_frame_count += 1
        else:
            self.hei_frame_flag = False
        self.video_frame_count += 1
        return
    
    
    def _start_end_point(self, cx, cy, distance):
        start_point = (cx-distance, cy-distance)
        end_point = (cx+distance, cy+distance)
        return start_point, end_point
    

    def square_box(self, image):
        if self.hei_frame_flag and self.results.multi_hand_landmarks:
            hands_type = []
            h,w,c = image.shape

            for hand in self.results.multi_handedness:
                hand_type=hand.classification[0].label
                hands_type.append(hand_type)
            
            i = 0
            for handLms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):

                    if id == 0: # 0 is the wrist id
                        wrist_cx, wrist_cy = int(lm.x * w), int(lm.y * h)

                    elif id == 9: # 0 is the middle finger metacarpophalangeal id
                        mf_mcp_cx, mf_mcp_cy = int(lm.x * w), int(lm.y * h)

                distance = int(1.25 * np.sqrt((wrist_cx-mf_mcp_cx)**2 + (wrist_cy-mf_mcp_cy)**2)) # 1.25 * Euclidian distance for hand size reference
                start_point, end_point = self._start_end_point(mf_mcp_cx, mf_mcp_cy, distance)

                if start_point >= (0,0) and end_point <= (w,h):
                    distance_to_border = min(mf_mcp_cx, w-mf_mcp_cx, mf_mcp_cy, h-mf_mcp_cy) # min value from mf_mcp to the border 
                    hand_label = hands_type[i]
                    min_box_size = 2*distance # minimum square box side length
                    max_box_size = 2*distance_to_border # maximum square box side length
                    self.add_to_buffer(image, mf_mcp_cx, mf_mcp_cy, min_box_size, max_box_size, hand_label)

                    if len(hands_type) == 1:
                        if hand_label == 'Left':
                            self.add_to_buffer([], -1, -1, -1, -1, 'Right')
                        elif hand_label == 'Right':
                            self.add_to_buffer([], -1, -1, -1, -1, 'Left')
                        
                i += 1
        return
    
    
    def add_to_buffer(self, image, cx, cy, min_box_size, max_box_size, hand):
        if hand == 'Left':
            self.left_image_buffer.appendleft(image)
            self.left_center_coordinates.appendleft([cx,cy])
            self.left_min_box_size_buffer.appendleft(min_box_size)
            self.left_max_box_size_buffer.appendleft(max_box_size)
        elif hand == 'Right':
            self.right_image_buffer.appendleft(image)
            self.right_center_coordinates.appendleft([cx,cy])
            self.right_min_box_size_buffer.appendleft(min_box_size)
            self.right_max_box_size_buffer.appendleft(max_box_size)


    def crop_square(self, image, cx, cy, square_size):
        start_point, end_point = self._start_end_point(cx, cy, int(square_size/2))
        cropped_image = image[start_point[1]:end_point[1], start_point[0]:end_point[0]]
        return cropped_image
    

    def image_averaging(self, label='', last_hei_flag=False, save=False, folder_path=os.getcwd(), file_name='', extension='jpg'):
        if self.first_hei_flag:
            first_hei = self.hei_frame_count + 1 == self.buffer_length
            subsequent_hei = False
        else:
            subsequent_hei = self.hei_frame_count == self.hei_overlap_frame
            first_hei = False

        if last_hei_flag:
            last_hei = self.first_hei_flag or (self.hei_frame_count >= int(0.5 * self.hei_overlap_frame))
        else:
            last_hei = False

        if first_hei or subsequent_hei or last_hei:
            self.first_hei_flag = False
            self.hei_frame_count = 0

            hand = 'Left'
            images = self.left_image_buffer
            coordinates = self.left_center_coordinates
            min_common_box_size = max(self.left_min_box_size_buffer)
            max_common_box_size_list = [x for x in self.left_max_box_size_buffer if x>0]
            if len(max_common_box_size_list) > 0:
                max_common_box_size = min(max_common_box_size_list)
            else:
                max_common_box_size = -1 

            if min_common_box_size <= max_common_box_size:
                avg_image = []            
                k = 0
                # for image in list(islice(images,0,len(images))):
                for i in range(self.buffer_length):
                    if len(images[i]) > 0 and -1 not in coordinates[i]:
                        if len(avg_image) == 0:
                            cx, cy = coordinates[i]
                            cropped_image = self.crop_square(images[i], cx, cy, min_common_box_size)
                            avg_image = cropped_image
                            k += 1
                        else:
                            cx, cy = coordinates[i]
                            cropped_image = self.crop_square(images[i], cx, cy, min_common_box_size)
                            alpha = 1.0/(k + 1)
                            beta = 1.0 - alpha
                            try:
                                avg_image = cv2.addWeighted(cropped_image, alpha, avg_image, beta, 0.0)
                            except:
                                print(f'cropped_image', cropped_image.shape)
                                print(f'avg_image', avg_image.shape)
                            k += 1   

                if save:
                    file_name = file_name + '_' + hand
                    folder_path_left = os.path.join(folder_path, hand, label)
                    self.save_image(avg_image, folder_path=folder_path_left, file_name=file_name, extension=extension)
                left_hei = avg_image
            else:
                left_hei = []


                
            hand = 'Right'
            images = self.right_image_buffer
            coordinates = self.right_center_coordinates
            min_common_box_size = max(self.right_min_box_size_buffer)
            max_common_box_size_list = [x for x in self.right_max_box_size_buffer if x>0]
            if len(max_common_box_size_list) > 0:
                max_common_box_size = min(max_common_box_size_list)
            else:
                max_common_box_size = -1
            

            if min_common_box_size <= max_common_box_size:
                avg_image = []            
                k = 0
                # for image in list(islice(images,0,len(images))):
                for i in range(self.buffer_length):
                    if len(images[i]) > 0 and -1 not in coordinates[i]:
                        if len(avg_image) == 0:
                            cx, cy = coordinates[i]
                            cropped_image = self.crop_square(images[i], cx, cy, min_common_box_size)
                            avg_image = cropped_image
                            k += 1
                        else:
                            cx, cy = coordinates[i]
                            cropped_image = self.crop_square(images[i], cx, cy, min_common_box_size)
                            alpha = 1.0/(k + 1)
                            beta = 1.0 - alpha
                            try:
                                avg_image = cv2.addWeighted(cropped_image, alpha, avg_image, beta, 0.0)
                            except:
                                print(f'cropped_image', cropped_image.shape)
                                print(f'avg_image', avg_image.shape)
                            k += 1   

                if save:
                    file_name = file_name + '_' + hand
                    folder_path_right = os.path.join(folder_path, hand, label)
                    self.save_image(avg_image, folder_path=folder_path_right, file_name=file_name, extension=extension)
                right_hei = avg_image
            else:
                right_hei = []

            return left_hei, right_hei
        else:
            return [[],[]]
        
        
        
    def save_image(self, image, file_name, extension='jpg', folder_path=os.getcwd()):
        if len(image)>0 and not os.path.exists(os.path.join(folder_path, file_name + '.' + extension)):
            cv2.imwrite(os.path.join(folder_path, file_name + '.' + extension), image)
        return
    

    def tracking(self, image, subpixel_layout='BGR'):
        if subpixel_layout == 'BGR':
            imageBGR = image
            imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif subpixel_layout == 'RGB':
            imageRGB = image
            imageBGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        self.hands_finder(imageRGB)
        self.square_box(imageBGR)
        return
    