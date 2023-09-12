import tensorflow as tf
import cv2
import numpy as np
import sys
sys.path.insert(1, './hand-tracking')
import handtracker

def main():
    model = tf.keras.models.load_model('saved_models/trainset_v4')

    class_names = ['abbonamento',
                    'abitare',
                    'acqua',
                    'affitto',
                    'allegare',
                    'allergia',
                    'ambulanza',
                    'amministrazione',
                    'ancona',
                    'andata']

    cap = cv2.VideoCapture(0)
    tracker = handtracker.HandTracker(hei_frame_step=4, hei_max_frames=10, hei_overlap=0)
    success = True
    normalization_layer = tf.keras.layers.Rescaling(1./255)

    while success == True:
        success, image = cap.read()
        if success:
            cv2.imshow("webcam", image)
            cv2.waitKey(50)
            tracker.tracking(image, subpixel_layout='BGR')   

            hei_left, hei_right = tracker.image_averaging(save=False)
            if len(hei_left) > 0:
                cv2.imshow("HEI left", hei_left)
                cv2.waitKey(500)
                hei_left = tf.image.resize(hei_left, [104,104])
                hei_left = np.expand_dims(hei_left, axis=0)
                print(f'Prediction: {class_names[int(np.argmax(model.predict(hei_left, verbose=0)))]}')
        
        if tracker.video_frame_count == 600:
            cap.release()
            success = False


if __name__ == "__main__":
    main()