import cv2
import mediapipe as mp

import numpy as np
from pathlib import Path
from collections import deque
from itertools import islice
from time import time_ns


# Hand tracker that stores all hand landmarks coordinates
class HandTracker3():
    def __init__(self, video_fps: int=0, sampling_rate: int=0, frame_step: int=2, max_duration: float=0,
                 overlap: float=0, max_frames: int=30, mode: str='IMAGE', max_hands: int=2,
                 detection_con: float=0.5, model_complexity: int=1, track_con: float=0.5) -> None:
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.model_complexity = model_complexity
        self.track_con = track_con
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands,self.model_complexity,
                                        self.detection_con, self.track_con)
        self.mp_drawing = mp.solutions.drawing_utils
        self.results=None

        if max_duration and sampling_rate:
            buffer_length = int(sampling_rate * max_duration)
        else:
            buffer_length = max_frames

        if video_fps and sampling_rate:
            frame_step = int(video_fps/sampling_rate)

        self.frame_step = frame_step
        self.buffer_length = buffer_length
        self.overlap_frame = int((1-overlap)*buffer_length)
        self.frame_count = 0
        self.time_series_frame_flag = True
        self.first_time_series_flag = True
        
        self.video_frame_count = 0

        self.left_hand_coordinates = deque(buffer_length*[[]], buffer_length)
        self.right_hand_coordinates = deque(buffer_length*[[]], buffer_length)


    def hands_finder(self,imageRGB):
        if self.video_frame_count % self.frame_step == 0:
            self.time_series_frame_flag = True
            self.results = self.hands.process(imageRGB)
            self.frame_count += 1
        else:
            self.time_series_frame_flag = False
        self.video_frame_count += 1
        return
    
    
    def get_coordinates(self, image_shape):
        if self.time_series_frame_flag and self.results.multi_hand_landmarks:
            hands_type = []
            h,w,c = image_shape

            # for hand in self.results.multi_handedness:
            #     hand_type=hand.classification[0].label
            #     hands_type.append(hand_type)

            image = np.zeros((h, w, 3), dtype = np.uint8)
            
            for handLms in self.results.multi_hand_landmarks:
                # Drawing Land Marks
                self.mp_drawing.draw_landmarks(
                    image, 
                    handLms, 
                    self.mp_hands.HAND_CONNECTIONS
                    )
                
            cv2.imwrite(f'hand-tracking/test/test{self.frame_count}.jpg', image)
                
            
        

            # i = 0
            # coordinates = 21*[[-1,-1]]
            # for handLms in self.results.multi_hand_landmarks:

            #     hand_label = hand_type[i]

            #     for id, lm in enumerate(handLms.landmark):

            #         cx , cy = int(lm.x * w), int(lm.y * h)                    
            #         coordinates[id] = [cx, cy]
                

            #     self.add_to_buffer(coordinates, hand_label)     
                
            #     if len(hands_type) == 1:
            #         if hand_label == 'Left':
            #             self.add_to_buffer(id, [], 'R')
            #         elif hand_label == 'Right':
            #             self.add_to_buffer(id, [], 'L')
                        
            #     i += 1
        return
    
    
    # def add_to_buffer(self, coordinates, hand):
    #     if hand == 'L':
    #         self.left_hand_coordinates.appendleft(coordinates)
    #     elif hand == 'R':
    #         self.right_hand_coordinates.appendleft(coordinates)


    # def return_time_series(self, label='', last_time_series_flag=False):
    #     if self.first_time_series_flag:
    #         first_hei = self.time_series_frame_flag + 1 == self.buffer_length
    #         subsequent_hei = False
    #     else:
    #         subsequent_hei = self.time_series_frame_flag == self.overlap_frame
    #         first_hei = False

    #     if last_time_series_flag:
    #         last_hei = self.first_time_series_flag or (self.time_series_frame_flag >= int(0.5 * self.overlap_frame))
    #     else:
    #         last_hei = False

    #     if first_hei or subsequent_hei or last_hei:
    #         self.first_time_series_flag = False
    #         self.frame_count = 0

    #         buffer = np.array(self.left_hand_coordinates)
    #         reference = buffer[0]

    #         relative_coordinates = dict()

    #         for hand in buffer:
    #             for landmark in range(len(hand)):
    #                 try:
    #                     relative_coordinates[str(landmark) + 'x'].append(hand[landmark][0] - reference[landmark][0])
    #                     relative_coordinates[str(landmark) + 'y'].append(hand[landmark][1] - reference[landmark][1])
    #                 except:
    #                     relative_coordinates[str(landmark) + 'x'] = [hand[landmark][0] - reference[landmark][0]]
    #                     relative_coordinates[str(landmark) + 'y'] = [hand[landmark][1] - reference[landmark][1]]

    #         return relative_coordinates
            

    
    
    def tracking(self, image, subpixel_layout='BGR'):
        if subpixel_layout == 'BGR':
            imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif subpixel_layout == 'RGB':
            imageRGB = image

        self.hands_finder(imageRGB)
        self.get_coordinates(imageRGB.shape)
        
        return
    
    
    
def main():
    video = "D:/Documentos/Polito/Thesis/Datasets/A3LIS-147_italian/trimmed-10/test/mdq_allergia (online-video-cutter.com).mp4"
    cap = cv2.VideoCapture(video)
    tracker = HandTracker3()
    print('Capturing started')


    while True:
        success,image = cap.read()

        if success:
            tracker.tracking(image)

            # print(tracker.return_time_series())

        else:
            cap.release()
            video_array = []
            images_path = Path('hand-tracking/test/')
            for file_name in images_path.glob('*.jpg'):
                img = cv2.imread(str(file_name))
                video_array.append(img)

            out = cv2.VideoWriter('hand-tracking/test/test.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 7, (len(img), len(img[0])))
            for i in range(len(video_array)):
                out.write(video_array[i])
            out.release()
            # print(tracker.left_hand_coordinates)
            break

        # cv2.imshow("Video", image)
        # cv2.waitKey(50)
        

if __name__ == "__main__":
    main()