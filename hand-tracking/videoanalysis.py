import cv2

file_name = "D:/Documentos/Polito/Thesis/Datasets/A3LIS-147_italian/fal_trimmed/fal_abbonamento.avi"

cap = cv2.VideoCapture(file_name)
fps = cap.get(cv2.CAP_PROP_FPS)
n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(f'fps = {fps} | Number of frames = {n_frames} | Duration (s) = {round(n_frames/fps,2)}')

