import cv2
import os

import handtracker

def main():

    # video_folder_path = "D:/Documentos/Polito/Thesis/Datasets/A3LIS-147_italian/trimmed-10/train/"
    # hei_folder_path = "D:/Documentos/Polito/Thesis/Datasets/A3LIS-147_italian/trimmed-10/hei-train/"
    video_folder_path = "D:/Documentos/Polito/Thesis/Datasets/A3LIS-147_italian/trimmed-10/test/"
    hei_folder_path = "D:/Documentos/Polito/Thesis/Datasets/A3LIS-147_italian/trimmed-10/hei-test/"
    hei_file_extension = 'jpg'

    for video_file_name in os.listdir(video_folder_path):

        hei_file_name = video_file_name.split(' ')[0]

        file_exists = (os.path.isfile(os.path.join(hei_folder_path, hei_file_name+'_left.'+hei_file_extension))
                       or os.path.isfile(os.path.join(hei_folder_path, hei_file_name+'_right.'+hei_file_extension)))

        if file_exists:
            pass
        else:
            tracker = handtracker.HandTracker(video_fps=25, hei_sampling_rate=12, hei_max_duration=4)
            cap = cv2.VideoCapture(os.path.join(video_folder_path, video_file_name))
            success = True
            while success == True:
                success, image = cap.read()
                if success:
                    tracker.tracking(image, subpixel_layout='BGR')         
            
                    
            hei_left = tracker.image_averaging('Left', save=True,folder_path=hei_folder_path,
                                               file_name=hei_file_name+'_left', extension=hei_file_extension)
            hei_right = tracker.image_averaging('Right', save=True, folder_path=hei_folder_path,
                                                file_name=hei_file_name+'_right', extension=hei_file_extension)


            if len(hei_left) > 0:
                cv2.imshow("HEI left", hei_left)
                cv2.waitKey(500)
            if len(hei_right) > 0:
                cv2.imshow("HEI right", hei_right)
                cv2.waitKey(500)

    # video_file_name = "fef_allegare (online-video-cutter.com).mp4"
    # hei_file_name = video_file_name.split(' ')[0]

    # tracker = handtracker.HandTracker(video_fps=25, hei_sampling_rate=12, hei_max_duration=4)
    # cap = cv2.VideoCapture(os.path.join(video_folder_path, video_file_name))
    # success = True
    # while success == True:
    #     success, image = cap.read()
    #     if success:
    #         tracker.tracking(image, subpixel_layout='BGR')         
    
            
    # hei_left = tracker.image_averaging('Left', save=True,folder_path=hei_folder_path,
    #                                     file_name=hei_file_name+'_left', extension=hei_file_extension)
    # hei_right = tracker.image_averaging('Right', save=True, folder_path=hei_folder_path,
    #                                     file_name=hei_file_name+'_right', extension=hei_file_extension)


    # if len(hei_left) > 0:
    #     cv2.imshow("HEI left", hei_left)
    #     cv2.waitKey(500)
    # if len(hei_right) > 0:
    #     cv2.imshow("HEI right", hei_right)
    #     cv2.waitKey(500)

if __name__ == "__main__":
    main()


