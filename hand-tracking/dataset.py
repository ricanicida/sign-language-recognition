import cv2
import os

import handtracker

LABELS = ['abbonamento', 'abitare', 'acqua', 'affitto', 'allegare',
          'allergia', 'ambulanza', 'amministrazione', 'ancona', 'andata']

def main():

    # video_folder_path = "D:/Documentos/Polito/Thesis/Datasets/A3LIS-147_italian/trimmed-10/train/"
    # hei_folder_path = "D:/Documentos/Polito/Thesis/Datasets/A3LIS-147_italian/trimmed-10/hei-train-v3/"
    video_folder_path = "D:/Documentos/Polito/Thesis/Datasets/A3LIS-147_italian/trimmed-10/test/"
    hei_folder_path = "D:/Documentos/Polito/Thesis/Datasets/A3LIS-147_italian/trimmed-10/hei-test-v3/"

    hei_file_extension = 'jpg'
    
    if not os.path.exists(hei_folder_path):
        os.mkdir(hei_folder_path)
    
    for hand in ['left', 'right']:
        path = os.path.join(hei_folder_path, hand)
        if not os.path.exists(path):
            os.mkdir(path)

        for label in LABELS:
            path = os.path.join(hei_folder_path, hand, label)
            if not os.path.exists(path):
                os.mkdir(path)

    for video_file_name in os.listdir(video_folder_path):

        hei_file_name = video_file_name.split(' ')[0]
        label = hei_file_name.split('_')[1]

        tracker = handtracker.HandTracker(video_fps=25, hei_sampling_rate=12, hei_max_duration=1, hei_overlap=0.25)
        cap = cv2.VideoCapture(os.path.join(video_folder_path, video_file_name))
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


