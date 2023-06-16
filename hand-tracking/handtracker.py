import cv2
import mediapipe as mp

import numpy as np


class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,modelComplexity=1,trackCon=0.5) -> None:
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.results=None

    def hands_finder(self,imageRGB):
        self.results = self.hands.process(imageRGB)
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
                for id, lm in enumerate(self.results.multi_hand_landmarks[0].landmark):

                    if id == 0: # 0 is the wrist id
                        wrist_cx, wrist_cy = int(lm.x * w), int(lm.y * h)

                    elif id == 9: # 0 is the middle finger metacarpophalangeal id
                        mf_mcp_cx, mf_mcp_cy = int(lm.x * w), int(lm.y * h)

                distance = int(1.25 * np.sqrt((wrist_cx-mf_mcp_cx)**2 + (wrist_cy-mf_mcp_cy)**2)) # 1.25 * Euclidian distance for hand size reference
                start_point = (mf_mcp_cx-distance, mf_mcp_cy+distance)
                end_point = (mf_mcp_cx+distance, mf_mcp_cy-distance)
                if start_point >= (0,0) and end_point <= (w,h) and draw:
                    cv2.rectangle(image, start_point, end_point, (0,255,0), 2)
                else:
                    mf_mcp_cx, mf_mcp_cy, distance = None, None, None
                
                hand_label = hands_type[i]
                print(hand_label)
                position_info[hand_label] = [mf_mcp_cx, mf_mcp_cy, distance]
                i += 1
            return position_info, image
        
        else:
            return None, image
    
    
def main():
    cap = cv2.VideoCapture(0)
    tracker = handTracker()

    while True:
        success,image = cap.read()
        imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        tracker.hands_finder(imageRGB)
        position_info, output_image = tracker.square_boxes_position(image)

        cv2.imshow("Video",output_image)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()