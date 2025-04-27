import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class BestEpochTracker(Callback):
    def __init__(self):
        super().__init__()
        self.best_epoch = None
        self.best_val_accuracy = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs and "val_accuracy" in logs:
            current_val_accuracy = logs["val_accuracy"]
            if current_val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = current_val_accuracy
                self.best_epoch = epoch + 1

class NeuralNetworkModel:
    def __init__(self, num_classes, dropout_rate=None, model_version=0):
        """
        Initialize the neural network model.
        
        Args:
            num_classes (int): Number of output classes.
            dropout_rate (float, optional): Dropout rate to apply before the final dense layer. Default is None.
        """
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model_version = model_version
        self.build_model()

    def build_model(self):
        """
        Build the CNN model with optional dropout.
        """
        print("==== Building model ====")
        if self.model_version == 0:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Rescaling(1./255),

                # Convolutional Layers
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                # Fully Connected Layer
                tf.keras.layers.Dense(128, activation='relu'),
            ])
        elif self.model_version == 1:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Rescaling(1./255),

                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(),

                tf.keras.layers.SeparableConv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(),

                tf.keras.layers.SeparableConv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(),

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
            ])
        elif self.model_version == 2:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Rescaling(1./255),

                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D(),

                # Residual Block
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
            ])

        # Add Dropout layer if dropout_rate is provided
        if self.dropout_rate is not None:
            self.model.add(tf.keras.layers.Dropout(self.dropout_rate))

        # Final Output Layer
        self.model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))

    def compile_model(self):
        print("==== Compiling model ====")
        self.model.compile(
            optimizer='adam',
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy'])

    def train(self, train_ds, epochs, val_ds=None, checkpoint_filepath=None):
        if val_ds is not None:
            print("==== Training and validating model ====")
            if checkpoint_filepath is None:
                self.checkpoint_filepath = "/temp/ckpt/checkpoint.model.keras"
            else:
                self.checkpoint_filepath = checkpoint_filepath
            model_checkpoint_callback = ModelCheckpoint(
                filepath=self.checkpoint_filepath,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True)

            epoch_tracker = BestEpochTracker()

            history = self.model.fit(
                train_ds,
                epochs=epochs,
                validation_data=val_ds,
                callbacks=[model_checkpoint_callback, epoch_tracker]
            )
            self.best_epoch = epoch_tracker.best_epoch
        else:
            print("==== Training model ====")
            history = self.model.fit(
                train_ds,
                epochs=epochs
            )
        
        return history

    def load_weights(self, checkpoint_filepath=None):
        if checkpoint_filepath is None:
            self.model.load_weights(self.checkpoint_filepath)
        else:
            self.model.load_weights(checkpoint_filepath)

    def evaluate(self, test_ds):
        accuracy = tf.keras.metrics.Accuracy()
        for x_batch, y_batch in test_ds:
            probabilities = self.model.predict(x_batch)
            predictions = np.argmax(probabilities, axis=-1)
            accuracy.update_state(y_batch, predictions)

        final_accuracy = accuracy.result().numpy()
        return final_accuracy

    def save_weights(self, model_path):
        self.model.save_weights(model_path)

    def load_saved_weights(self, model_path):
        self.model.load_weights(model_path)


class CombinedModel:
    def __init__(self, model1, model2, model1_class_names, model2_class_names, weight1=1, weight2=1, extra_weight2=4, top_for_extra=3):
        """
        Initializes the CombinedModel with two NeuralNetworkModel instances.

        Args:
            model1 (NeuralNetworkModel): First model instance.
            model2 (NeuralNetworkModel): Second model instance.
            weight1 (float): Weight for model1 predictions.
            weight2 (float): Weight for model2 predictions.
            extra_weight2 (float): Extra weight for top predictions in model2.
            top_for_extra (int): Number of top predictions to apply extra weight.
        """
        self.model1 = model1.model
        self.model2 = model2.model
        self.model1_class_names = model1_class_names
        self.model2_class_names = model2_class_names
        self.weight1 = weight1
        self.weight2 = weight2
        self.extra_weight2 = extra_weight2
        self.top_for_extra = top_for_extra

    def average_model_predictions(self, test_ds):
        """
        Averages predictions from both models while handling shared labels.
        """
        shared_labels = [label for label in self.model1_class_names if label in self.model2_class_names]
        shared_indices_model1 = [self.model1_class_names.index(label) for label in shared_labels]
        shared_indices_model2 = [self.model2_class_names.index(label) for label in shared_labels]
        
        predictions, predictions1, predictions2, true_labels = [], [], [], []
        pred1_size = len(self.model1_class_names)
        pred2_size = len(self.model2_class_names)
        
        for inputs, labels in test_ds:
            inputs1, inputs2 = inputs
            outputs1 = self.model1.predict(inputs1, verbose=0)
            outputs2 = self.model2.predict(inputs2, verbose=0)
            
            predictions1.append(outputs1)
            predictions2.append(outputs2)
            
            preds1 = outputs1.copy()
            preds2 = outputs2.copy()
            
            for i in range(len(inputs2)):
                if np.sum(inputs2[i]) == 0:
                    preds2[i] = np.zeros(pred2_size)
                elif (
                    np.argmax(preds2[i]) == self.model2_class_names.index('idle')
                    and np.sum(inputs1[i]) != 0
                ):
                    preds2[i] = np.zeros(pred2_size)

                if (
                    self.model2_class_names[np.argmax(preds2[i])]
                    in [
                    self.model1_class_names[j]
                    for j in np.argpartition(preds1[i], -self.top_for_extra)[-self.top_for_extra:]
                    ]
                    and np.sum(inputs1[i]) != 0
                ):
                    preds2[i] *= self.extra_weight2

                if np.sum(inputs1[i]) == 0:
                    preds1[i] = np.zeros(pred1_size)
            
            averaged_preds = np.zeros_like(preds1)
            averaged_preds += self.weight1 * preds1
            preds2_shared = tf.gather(preds2, indices=shared_indices_model2, axis=-1).numpy()
            averaged_preds[:, shared_indices_model1] += self.weight2 * preds2_shared
            
            for i in range(len(averaged_preds)):
                averaged_preds[i] /= np.sum(averaged_preds[i])
            
            predictions.append(averaged_preds)
            true_labels.append(labels.numpy())
        
        self.predictions = np.vstack(predictions)
        self.true_labels = np.concatenate(true_labels)
        self.predictions1 = np.vstack(predictions1)
        self.predictions2 = np.vstack(predictions2)

    def average_true_prob(self):
        # Extract the predicted probabilities for the true class labels
        true_probs = [self.predictions[i][label] for i, label in enumerate(self.true_labels)]
        # Compute the average predicted probability
        avg_prob = np.mean(true_probs)
        return avg_prob

    def top_n_accuracy(self, n):
        correct = 0
        for i, probs in enumerate(self.predictions):
            top_n_preds = np.argsort(probs)[-n:]  # Get indices of top-n predictions
            if self.true_labels[i] in top_n_preds:
                correct += 1
        return correct / len(self.true_labels)
    
    def generate_confusion_matrix(self, file_path): 
        # Compute predicted labels
        predicted_labels = np.argmax(self.predictions, axis=1)

        # Generate confusion matrix
        cm = confusion_matrix(self.true_labels, predicted_labels)

        # Plot high-resolution confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.model1_class_names,
                    yticklabels=self.model1_class_names,
                    annot_kws={"size": 16})  # Increase numbers inside the cells

        # Set axis labels and title with larger font
        plt.xlabel('Predicted Labels', fontsize=18)
        plt.ylabel('True Labels', fontsize=18)
        plt.title('Confusion Matrix', fontsize=20)

        # Increase tick labels font size
        plt.xticks(fontsize=18, rotation=45, ha='right')
        plt.yticks(fontsize=18, rotation=0)

        plt.tight_layout()
        
        # Save to file
        plt.savefig(file_path, dpi=300)
        plt.close()

    def evaluate(self, test_ds):
        self.average_model_predictions(test_ds)
        predicted_classes = np.argmax(self.predictions, axis=1)
        accuracy = np.mean(predicted_classes == self.true_labels)
        print("Inference Accuracy:", accuracy)
        top_2_acc = self.top_n_accuracy(2)
        print("Top 2 Accuracy:", top_2_acc)
        top_3_acc = self.top_n_accuracy(3)
        print("Top 3 Accuracy:", top_3_acc)
        avg_prob = self.average_true_prob()
        print("Average probability of true label:", avg_prob)

    def show_predictions(self, top=3):
        """
        Displays top predictions for both models and the combined model.
        """
        
        test_array = list(zip(self.predictions1, self.predictions2, self.predictions, self.true_labels))
        
        for i, (pred1, pred2, avg_pred, true_label_idx) in enumerate(test_array):
            pred1, pred2, avg_pred = np.array(pred1), np.array(pred2), np.array(avg_pred)
            
            top_idx_p1 = np.argsort(pred1)[-top:][::-1]
            top_prob_p1 = pred1[top_idx_p1]
            top_idx_p2 = np.argsort(pred2)[-top:][::-1]
            top_prob_p2 = pred2[top_idx_p2]
            top_idx_avg = np.argsort(avg_pred)[-top:][::-1]
            top_prob_avg = avg_pred[top_idx_avg]
            
            true_label_name = self.model1_class_names[true_label_idx]
            prediction_label = self.model1_class_names[np.argmax(avg_pred)]
            
            print(f"Sample {i + 1}:")
            print("  True Label:", true_label_name)
            print("  Prediction:", prediction_label)
            print("  Model 1 (left) Predictions:")
            for idx, prob in zip(top_idx_p1, top_prob_p1):
                print(f"    {self.model1_class_names[idx]}: {prob * 100:.1f}%")
            print("  Model 2 (right) Predictions:")
            for idx, prob in zip(top_idx_p2, top_prob_p2):
                if idx < len(self.model2_class_names):
                    print(f"    {self.model2_class_names[idx]}: {prob * 100:.1f}%")
                else:
                    print(f"    Unknown: {prob * 100:.1f}%")
            print("  Averaged Predictions:")
            for idx, prob in zip(top_idx_avg, top_prob_avg):
                print(f"    {self.model1_class_names[idx]}: {prob * 100:.1f}%")
            print("\n" + "-" * 50 + "\n")

class TransferLearningModel:
    def __init__(self, base_model, num_classes, dense_units=128, train_base=False, dropout_rate=0.5):
        """
        Initializes a transfer learning model using a pre-trained base model.

        Args:
            base_model (tf.keras.Model): Pre-trained base model (e.g., ResNet152).
            num_classes (int): Number of output classes.
            dense_units (int): Number of neurons in the added dense layer.
            train_base (bool): Whether to fine-tune the base model.
            dropout_rate (float): Dropout rate for regularization.
        """
        self.base_model = base_model
        self.num_classes = num_classes
        self.dense_units = dense_units
        self.train_base = train_base
        self.dropout_rate = dropout_rate

        self.build_model()

    def build_model(self):
        """
        Builds the fine-tuned model by adding custom layers.
        """
        # Freeze or unfreeze base model layers
        self.base_model.trainable = self.train_base
        
        # Add custom layers on top of the base model
        x = tf.keras.layers.GlobalAveragePooling2D()(self.base_model.output)  # Pooling layer
        x = tf.keras.layers.Dense(self.dense_units, activation='relu')(x)  # Fully connected layer
        if self.dropout_rate:
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)  # Dropout for regularization
        output = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)  # Classification layer

        # Create the final model
        self.model = tf.keras.Model(inputs=self.base_model.input, outputs=output)

    def compile_model(self, learning_rate=0.0001):
        """
        Compiles the model with an optimizer, loss function, and evaluation metrics.
        """
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )

    def train(self, train_ds, val_ds=None, epochs=10, checkpoint_filepath=None):
        """
        Trains the model using the provided dataset.

        Args:
            train_ds: Training dataset.
            val_ds: Validation dataset.
            epochs (int): Number of training epochs.
        """
        if val_ds is not None:
            print("==== Training and validating model ====")
            if checkpoint_filepath is None:
                self.checkpoint_filepath = "/temp/ckpt/checkpoint.model.keras"
            else:
                self.checkpoint_filepath = checkpoint_filepath
            
            model_checkpoint_callback = ModelCheckpoint(
                filepath=self.checkpoint_filepath,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True)

            epoch_tracker = BestEpochTracker()
            history = self.model.fit(
                train_ds, validation_data=val_ds, epochs=epochs,
                callbacks=[model_checkpoint_callback, epoch_tracker])
            
            self.best_epoch = epoch_tracker.best_epoch
        else:
            history = self.model.fit(
                train_ds, epochs=epochs)

        return history

    def evaluate(self, test_ds):
        """
        Evaluates the model on the test dataset.
        """
        return self.model.evaluate(test_ds)

    def save_model(self, filepath):
        """
        Saves the trained model.
        """
        self.model.save(filepath)

    def load_model(self, filepath):
        """
        Loads a saved model.
        """
        self.model = tf.keras.models.load_model(filepath)
