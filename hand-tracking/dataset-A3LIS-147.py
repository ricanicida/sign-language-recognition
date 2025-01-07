import cv2
import os
import glob
import logging 

import handtracker

# LABELS = ['abbonamento', 'abitare', 'acqua', 'affitto', 'allegare',
#           'allergia', 'ambulanza', 'amministrazione', 'ancona', 'andata']

LABELS = ['abitare', 'acqua', 'affitto', 'banca', 'caldo', 'casa', 'cibo', 'data',
          'freddo', 'interprete', 'inviare', 'lingua', 'litro', 'mangiare', 'posta', 'telefono', 'idle']

def main():

    # creating the logger object
    logger = logging.getLogger() 
    logging.basicConfig(level=logging.INFO) 

    # video_folder_path = "D:/Documentos/Polito/Thesis/Datasets/A3LIS-147_italian/trimmed-life/videos/"
    # hei_folder_path = "D:/Documentos/Polito/Thesis/Datasets/A3LIS-147_italian/trimmed-life/hei-videos/"
    video_folder_path = "D:/Documentos/Polito/Thesis/Datasets/A3LIS-147_italian/trimmed-life/augmented/usual_colors/videos-naturalwhite/"
    hei_folder_path = "D:/Documentos/Polito/Thesis/Datasets/A3LIS-147_italian/trimmed-life/augmented/usual_colors/hei-videos-naturalwhite-v1/"

    hei_file_extension = 'jpg'
    
    if not os.path.exists(hei_folder_path):
        os.mkdir(hei_folder_path)

        logging.info("Create HEI folder")
    
    for hand in ['left', 'right']:
        path = os.path.join(hei_folder_path, hand)
        if not os.path.exists(path):
            os.mkdir(path)

        for label in LABELS:
            path = os.path.join(hei_folder_path, hand, label)
            if not os.path.exists(path):
                os.mkdir(path)
    
    logging.info("Create left and right, and labels folders")

    for video_file_name in os.listdir(video_folder_path):

        hei_file_name = video_file_name.split('.')[0]
        label = hei_file_name.split('_')[1]

        left_files = glob.glob(os.path.join(hei_folder_path, 'left', label, f"*_{hei_file_name}_Left.{hei_file_extension}"))
        right_files = glob.glob(os.path.join(hei_folder_path, 'right', label, f"*_{hei_file_name}_Right.{hei_file_extension}"))
        
        if  not left_files and not right_files:

            tracker = handtracker.HandTracker(video_fps=25, hei_sampling_rate=12, hei_max_duration=4, hei_overlap=0)
            cap = cv2.VideoCapture(os.path.join(video_folder_path, video_file_name))
            success = True
            while success == True:
                success, image = cap.read()
                if success:
                    tracker.tracking(image, subpixel_layout='BGR')         
            
                    hei_left, hei_right = tracker.image_averaging(label=label, save=True,folder_path=hei_folder_path,
                                                        file_name=hei_file_name, extension=hei_file_extension)
                    
                    if len(hei_left) > 0:
                        # cv2.imshow("HEI left", hei_left)
                        # cv2.waitKey(500)
                        # print(".", end=" ")
                        logging.info("Left")
                    if len(hei_right) > 0:
                        # cv2.imshow("HEI right", hei_right)
                        # cv2.waitKey(500)
                        # print(".", end=" ")
                        logging.info("Right")
            
                
            hei_left, hei_right = tracker.image_averaging(label=label, last_hei_flag=True,save=True,folder_path=hei_folder_path,
                                                file_name=hei_file_name, extension=hei_file_extension)
            
            if len(hei_left) > 0:
                # cv2.imshow("HEI left", hei_left)
                # cv2.waitKey(500)
                # print(".", end=" ")
                logging.info("Left")
            if len(hei_right) > 0:
                # cv2.imshow("HEI right", hei_right)
                # cv2.waitKey(500)
                # print(".", end=" ")
                logging.info("Right")
        
        else:
            # print(f"{hei_file_name} already exists.")
            logging.info(f"{hei_file_name} already exists")

    logging.info("Successful run")

if __name__ == "__main__":
    main()


