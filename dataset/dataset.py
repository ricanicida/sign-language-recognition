import tensorflow as tf
import pathlib
import logging

class Dataset:
    def __init__(self, train_folder_path, test_folder_path, train_batch_size, test_batch_size, validation_split=0.18, img_height=100, img_width=100):
        self.train_folder_path = pathlib.Path(train_folder_path)
        self.test_folder_path = pathlib.Path(test_folder_path)
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.validation_split = validation_split
        self.img_height = img_height
        self.img_width = img_width

        print("==== Loading train and validation data ====")
        self.train_ds, self.val_ds = self.load_train_data()
        print("==== Loading test data ====")
        self.test_ds = self.load_test_data()
        self.class_names = self.train_ds.class_names
        self.num_classes = len(self.class_names)
        print("==== Preparing datasets ====")
        self.prepare_datasets()

    def load_train_data(self):
        return tf.keras.utils.image_dataset_from_directory(
            self.train_folder_path,
            validation_split=self.validation_split,
            subset="both",
            seed=42,
            image_size=(self.img_height, self.img_width),
            batch_size=self.train_batch_size,
            shuffle=True)

    def load_test_data(self):
        return tf.keras.utils.image_dataset_from_directory(
            self.test_folder_path,
            image_size=(self.img_height, self.img_width),
            batch_size=self.test_batch_size)

    def prepare_datasets(self):
        AUTOTUNE = tf.data.AUTOTUNE
        self.train_ds = self.train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        self.test_ds = self.test_ds.cache().prefetch(buffer_size=AUTOTUNE)

        # Merge train and validation datasets
        self.train_val_ds = self.train_ds.concatenate(self.val_ds)
        self.train_val_ds = self.train_val_ds.cache().prefetch(buffer_size=AUTOTUNE)

class CombinedTestDataset:
    def __init__(self, left_hand_folder, right_hand_folder, img_height, img_width, batch_size):
        self.left_hand_folder = pathlib.Path(left_hand_folder)
        self.right_hand_folder = pathlib.Path(right_hand_folder)
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        
        self.test_ds = self.prepare_dataset()
    
    def get_image_prefixes_and_labels(self, folder_path):
        image_data = []
        for label_folder in folder_path.iterdir():
            if label_folder.is_dir():
                label = label_folder.name  # Folder name as the label
                for image_file in label_folder.iterdir():
                    if image_file.is_file():
                        file_name = "_".join(image_file.stem.split("_")[:-1])
                        image_data.append((file_name, str(image_file), label))
        return image_data
    
    def preprocess_image(self, image_path):
        if image_path is None:
            return tf.zeros((self.img_height, self.img_width, 3), dtype=tf.int32)
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(image, [self.img_height, self.img_width])
        image = tf.cast(image, tf.int32)
        return image
    
    def create_dataset(self, data):
        def generator():
            for left_path, right_path, label in data:
                yield (self.preprocess_image(left_path), self.preprocess_image(right_path)), self.label_to_index[label]
        
        return tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                (
                    tf.TensorSpec(shape=(self.img_height, self.img_width, 3), dtype=tf.int32),
                    tf.TensorSpec(shape=(self.img_height, self.img_width, 3), dtype=tf.int32),
                ),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            )
        )
    
    def prepare_dataset(self):
        left_images = self.get_image_prefixes_and_labels(self.left_hand_folder)
        right_images = self.get_image_prefixes_and_labels(self.right_hand_folder)

        left_dict = {prefix: (path, label) for prefix, path, label in left_images}
        right_dict = {prefix: (path, label) for prefix, path, label in right_images}

        aligned_data = []
        for prefix in set(left_dict.keys()).union(right_dict.keys()):
            left_info = left_dict.get(prefix, (None, None))
            right_info = right_dict.get(prefix, (None, None))
            aligned_data.append((left_info[0], right_info[0], left_info[1] or right_info[1]))
        
        unique_labels = sorted(set(label for _, _, label in aligned_data))
        self.label_to_index = {label: index for index, label in enumerate(unique_labels)}

        return self.create_dataset(aligned_data).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
