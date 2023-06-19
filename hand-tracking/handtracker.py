import cv2
import mediapipe as mp

import numpy as np
import os
from collections import deque


class HandTracker():
    def __init__(self, mode=False, max_hands=2, detection_con=0.5,model_complexity=1,track_con=0.5) -> None:
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.model_complexity = model_complexity
        self.track_con = track_con
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands,self.model_complexity,
                                        self.detection_con, self.track_con)
        self.results=None
        self.left_image_buffer = deque(5*[[]], 5)
        self.right_image_buffer = deque(5*[[]], 5)
        self.max_box_side = 0

    def hands_finder(self,imageRGB):
        self.results = self.hands.process(imageRGB)
        return
    
    def update_max_box_side(self):
        self.max_box_side = max([len(x) for x in self.left_image_buffer])
        return
    
    def square_boxes_position(self, image, draw=True):
        if self.results.multi_hand_landmarks:
            hands_type = []
            position_info = {}
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
                start_point = (mf_mcp_cx-distance, mf_mcp_cy-distance)
                end_point = (mf_mcp_cx+distance, mf_mcp_cy+distance)
                if start_point >= (0,0) and end_point <= (w,h) and draw:
                    cv2.rectangle(image, start_point, end_point, (0,255,0), 2)
                    hand_label = hands_type[i]
                    position_info[hand_label] = [start_point, end_point]
                
                i += 1
            return position_info, image
        
        else:
            return None, image
    
    def crop_hand(self, image, position_info, hand):
        if position_info and hand in position_info.keys():
            start_point, end_point = position_info[hand]
            cropped_image = image[start_point[1]:end_point[1], start_point[0]:end_point[0]]
            return cropped_image
        else:
            return []
        
    def add_to_buffer(self, image, hand):
        if hand == 'Left':
            self.left_image_buffer.appendleft(image)
        else:
            self.right_image_buffer.appendleft(image)

    def image_averaging(self, hand):
        if hand == 'Left':
            images = self.left_image_buffer
        else:
            images = self.right_image_buffer

        if len(images[-1]) > 0:
            dim = (self.max_box_side, self.max_box_side)
            avg_image = images[0]
            avg_image = cv2.resize(avg_image, dim, interpolation = cv2.INTER_AREA)
            for i in range(1, len(images)):
                image = cv2.resize(images[i], dim, interpolation = cv2.INTER_AREA)
                if i == 0:
                    pass
                else:
                    alpha = 1.0/(i + 1)
                    beta = 1.0 - alpha
                    avg_image = cv2.addWeighted(image, alpha, avg_image, beta, 0.0)
            return avg_image
        
    def save_image(self, image, file_name, folder_path=os.getcwd()):
        cv2.imwrite(os.path.join(folder_path, file_name), image)
        return
    
def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()

    i = 0
    while True:
        success,image = cap.read()
        imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        tracker.hands_finder(imageRGB)
        position_info, output_image = tracker.square_boxes_position(image)

        left_crop = tracker.crop_hand(image, position_info, 'Left')
        if len(left_crop) > 0:
            cv2.imshow("Cropped", left_crop)
            cv2.waitKey(1000)
            # tracker.save_image(left_crop, 'left-hand.jpg')
            tracker.add_to_buffer(left_crop, 'Left')
            tracker.update_max_box_side()
            i += 1
        if i == 5:
            print(tracker.max_box_side)
            hei = tracker.image_averaging('Left')
            if len(hei) > 0:
                cv2.imshow("HEI", hei)
                cv2.waitKey(0)
            break

        cv2.imshow("Video", output_image)
        cv2.waitKey(200)
        

if __name__ == "__main__":
    main()