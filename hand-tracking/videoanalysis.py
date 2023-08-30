import cv2
import os
import pandas as pd

SIGNERS = ['fal', 'fef', 'fsf', 'mdp', 'mdq', 'mic', 'mmr', 'mrla', 'mrlb', 'msf']

with open('SIGNS.txt') as f:
    SIGNS = f.readlines()
f.close()
SIGNS = [x.strip('\n') for x in SIGNS]

dataset_path = "D:/Documentos/Polito/Thesis/Datasets/A3LIS-147_italian/trimmed-10/"
csv_file_path = "D:/Documentos/Polito/Thesis/Datasets/A3LIS-147_italian/trimmed-10.csv"

df = pd.DataFrame(columns=['signer_id', 'name', 'sign_id', 'label', 'test_set', 'seconds'])

for video_name in os.listdir(os.path.join(dataset_path, 'train/')):
    name, label = video_name.split(' ')[0].split('_')
    signer_id = SIGNERS.index(name)
    sign_id = SIGNS.index(label)
    test_set = False

    cap = cv2.VideoCapture(os.path.join(dataset_path, 'train/', video_name))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f'fps = {fps} | Number of frames = {n_frames} | Duration (s) = {round(n_frames/fps,3)}')
    seconds = round(n_frames/fps, 3)

    df.loc[len(df.index)] = [signer_id, name, sign_id, label, test_set, seconds]

for video_name in os.listdir(os.path.join(dataset_path, 'test/')):
    name, label = video_name.split(' ')[0].split('_')
    signer_id = SIGNERS.index(name)
    sign_id = SIGNS.index(label)
    test_set = True

    cap = cv2.VideoCapture(os.path.join(dataset_path, 'test/', video_name))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f'fps = {fps} | Number of frames = {n_frames} | Duration (s) = {round(n_frames/fps,3)}')
    seconds = round(n_frames/fps, 3)

    df.loc[len(df.index)] = [signer_id, name, sign_id, label, test_set, seconds]

df.to_csv(csv_file_path)