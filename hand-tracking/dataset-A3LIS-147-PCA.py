import os
import cv2
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

def extract_pca_features_from_images(root_dir, pca_components=50, output_csv='pca_features.csv', image_size=(100, 100)):
    image_paths = []
    labels = []
    flat_images = []

    # Traverse subfolders
    for label in os.listdir(root_dir):
        label_folder = os.path.join(root_dir, label)
        if not os.path.isdir(label_folder):
            continue
        for file in os.listdir(label_folder):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(label_folder, file)
                image = cv2.imread(file_path)
                if image is None:
                    continue
                image = cv2.resize(image, image_size)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                flat_images.append(image.flatten())
                image_paths.append(file)
                labels.append(label)

    # Apply PCA
    pca = PCA(n_components=pca_components)
    pca_features = pca.fit_transform(flat_images)

    # Construct DataFrame
    df = pd.DataFrame(pca_features, columns=[f'pca_{i}' for i in range(pca_components)])
    df.insert(0, 'filename', image_paths)
    df.insert(1, 'label', labels)

    # Save CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved PCA features to {output_csv}")
    return

def main():
    train_test = "train"
    version = "v2"
    hei_folder_path = f"D:/Documentos/Polito/Thesis/Datasets/A3LIS-147_italian/trimmed-life/hei-videos-{train_test}-seed59-{version}/reduced/left"
    pca_csv_name = f"D:/Documentos/Polito/Thesis/Datasets/A3LIS-147_italian/trimmed-life/pca-hei-seed59-{version}/reduced/left_{train_test}_pca.csv"

    extract_pca_features_from_images(hei_folder_path, output_csv=pca_csv_name)

if __name__ == "__main__":
    main()


