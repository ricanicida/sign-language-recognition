import cv2
import os
from pathlib import Path

import handtracker

LABELS = ['abbonamento', 'abitare', 'acqua', 'affitto', 'allegare',
          'allergia', 'ambulanza', 'amministrazione', 'ancona', 'andata']

def main():

    video_folder_path = Path("D:/Documentos/Polito/Thesis/Datasets/MS-ASL/dataset_10/")
    hei_folder_path = Path("D:/Documentos/Polito/Thesis/Datasets/MS-ASL/dataset_10_HEI_v2/")

    hei_file_extension = 'jpg'
    
    for hand in ['left', 'right']:
        (hei_folder_path / hand).mkdir(parents=True, exist_ok=True)


    for video_file in video_folder_path.glob('*/*.mp4'):

        hei_file_name = video_file.name.split('.')[0]
        label = hei_file_name.split('-')[0]

        for hand in ['left', 'right']:
            (hei_folder_path / hand / label).mkdir(parents=True, exist_ok=True)

        left = (hei_folder_path / 'left' / label / (hei_file_name + '_' + 'Left' + '.' + hei_file_extension)).exists()
        right = (hei_folder_path / 'right' / label / (hei_file_name + '_' + 'Right' + '.' + hei_file_extension)).exists()

        if not (left or right):

            print(video_file)

            tracker = handtracker.HandTracker(hei_frame_step=2, hei_max_duration=1, hei_overlap=0.5)
            cap = cv2.VideoCapture(str(video_file))
            success = True
            while success == True:
                success, image = cap.read()
                if success:
                    tracker.tracking(image, subpixel_layout='BGR')         
            
                    hei_left, hei_right = tracker.image_averaging(label=label, save=True,folder_path=hei_folder_path,
                                                        file_name=hei_file_name, extension=hei_file_extension)
                    
                    if len(hei_left) > 0:
                        cv2.imshow("HEI left", hei_left)
                        cv2.waitKey(500)
                    if len(hei_right) > 0:
                        cv2.imshow("HEI right", hei_right)
                        cv2.waitKey(500)
            
                
            hei_left, hei_right = tracker.image_averaging(label=label, last_hei_flag=True,save=True,folder_path=hei_folder_path,
                                                file_name=hei_file_name, extension=hei_file_extension)
            
            if len(hei_left) > 0:
                cv2.imshow("HEI left", hei_left)
                cv2.waitKey(500)
            if len(hei_right) > 0:
                cv2.imshow("HEI right", hei_right)
                cv2.waitKey(500)


    # video_file_name = "fal_acqua (online-video-cutter.com).mp4"
    # hei_file_name = video_file_name.split(' ')[0]

    # tracker = handtracker.HandTracker(video_fps=25, hei_sampling_rate=12, hei_max_duration=1, hei_overlap=0.25)
    # cap = cv2.VideoCapture(os.path.join(video_folder_path, video_file_name))
    # success = True
    # while success == True:
    #     success, image = cap.read()
    #     if success:
    #         cv2.imshow("image", image)
    #         cv2.waitKey(200)
    #         tracker.tracking(image, subpixel_layout='BGR')         
    
    #         hei_left, hei_right = tracker.image_averaging(save=True,folder_path=hei_folder_path,
    #                                             file_name=hei_file_name, extension=hei_file_extension)
            
    #         if len(hei_left) > 0:
    #             cv2.imshow("HEI left", hei_left)
    #             cv2.waitKey(500)
    #         if len(hei_right) > 0:
    #             cv2.imshow("HEI right", hei_right)
    #             cv2.waitKey(500)
    
        
    # hei_left, hei_right = tracker.image_averaging(last_hei_flag=True,save=True,folder_path=hei_folder_path,
    #                                     file_name=hei_file_name, extension=hei_file_extension)
    
    # if len(hei_left) > 0:
    #     cv2.imshow("HEI left", hei_left)
    #     cv2.waitKey(500)
    # if len(hei_right) > 0:
    #     cv2.imshow("HEI right", hei_right)
    #     cv2.waitKey(500)

if __name__ == "__main__":
    main()