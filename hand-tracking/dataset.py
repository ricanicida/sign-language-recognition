import cv2

import handtracker

def main():
    file_name = "D:/Documentos/Polito/Thesis/Datasets/A3LIS-147_italian/fal_trimmed/fal_abbonamento.avi"
    # file_name = "D:/Documentos/Polito/Thesis/Datasets/A3LIS-147_italian/fal_trimmed/fal_abitare.avi"
    tracker = handtracker.HandTracker(video_fps=25, hei_sampling_rate=12, hei_max_duration=4)
    cap = cv2.VideoCapture(file_name)
    success = True
    while success == True:
        success, image = cap.read()
        if success:
            tracker.tracking(image, subpixel_layout='BGR')
            
    hei_left = tracker.image_averaging('Left', save=False)
    hei_right = tracker.image_averaging('Right', save=False)
    if len(hei_left) > 0:
        cv2.imshow("HEI left", hei_left)
        cv2.waitKey(0)
    if len(hei_right) > 0:
        cv2.imshow("HEI right", hei_right)
        cv2.waitKey(0)



if __name__ == "__main__":
    main()