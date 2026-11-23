# Casting Defect Detection Using CNN

## Overview

This project focuses on detecting defects in industrial casting products using a Convolutional Neural Network (CNN).

The goal is to classify casting images into two categories:

* **OK:** The casting product has no visible defects.

* **Defective:** The casting product contains defects.

I built this model using TensorFlow and Keras and trained it on a casting product image dataset. The project covers the complete workflow, including data preprocessing, model building, training, and evaluation.

## Dataset

I used the [Casting Product Image Dataset](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) from Kaggle.

The dataset contains grayscale images of casting products divided into two classes (0 = ok_front, 1 = def_front). Here is the breakdown:

* **OK:** 2,875 training images | 262 testing images

* **Defective:** 3,758 training images | 453 testing images

* **Total:** 6,633 training images | 715 testing images

The original training data (6,633 images) was further divided into training and validation sets using an 80:20 split, resulting in 5,306 training images and 1,327 validation images. I used stratified splitting to maintain the same class distribution in both sets.

## Project Structure

```
casting_defect_cnn/
│
├── data/
│   └── casting_data/
│       ├── train/
│       │   ├── ok_front/
│       │   └── def_front/
│       └── test/
│           ├── ok_front/
│           └── def_front/
│
├── models/
│   └── casting_cnn.keras
│
├── results/
│   ├── accuracy.png
│   └── loss.png
│
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── main.py
└── requirements.txt

```

## Data Preprocessing

Before training the model, I performed the following preprocessing steps:

* Resized all images to 224 × 224 pixels.

* Converted images to grayscale to reduce the input dimensions.

* Normalized pixel values to the range of 0 to 1.

* Split the training data into training and validation sets.

* Used TensorFlow's `tf.data` pipeline to load, batch, and process images efficiently.

The batch size was set to 32.

## Model Architecture

I built a CNN using TensorFlow and Keras.

The model consists of four convolutional blocks with 32, 64, 128, and 256 filters respectively. Each block contains two convolutional layers followed by batch normalization and ReLU activation. Max pooling is used to reduce the spatial dimensions after each block.

After the convolutional layers, I used Global Average Pooling, followed by a dropout layer and a final sigmoid output layer for binary classification. The final architecture has a total of 1,174,817 parameters.

## Model Training

The model was trained using the following configuration:

* Optimizer: Adam

* Learning Rate: 0.001

* Loss Function: Binary Crossentropy

* Batch Size: 32

* Epochs: 10

* Output Activation: Sigmoid

Since the dataset contains an imbalance between the two classes, I calculated class weights and used them during training (`{0: 1.153, 1: 0.883}`) to give more importance to the minority class.

I also used the following callbacks:

* **Early Stopping:** Stops training when the validation loss stops improving.

* **Model Checkpoint:** Saves the model with the best validation performance (best weights were restored from epoch 9).

* **ReduceLROnPlateau:** Reduces the learning rate when the validation loss stops improving.

The training and validation accuracy and loss graphs are saved to the `results/` directory.

## Evaluation

I evaluated the model using the following metrics:

* Accuracy

* Precision

* Recall

* F1-score

* ROC-AUC

I also plotted the confusion matrix and ROC curve to understand the model's performance.

Instead of using the default classification threshold of 0.5, I experimented with different thresholds using the validation set and selected the threshold that gave the best F1-score. The optimal validation threshold was found to be `0.1`, which was then used to evaluate the model on the test set.

## Results

The model achieved the following results on the test dataset at the `0.1` threshold:

* **Accuracy:** 99.86%

* **Precision:** 99.78%

* **Recall:** 100.00%

* **F1-score:** 99.89%

* **ROC-AUC:** 1.00

### Confusion Matrix

```
[[261   1]
 [  0 453]]

```

The model correctly classified 714 out of 715 test images. One non-defective image was incorrectly classified as defective.

## How to Run

Clone the repository:

```
git clone https://github.com/rangasaivenkat/castingdetection-using-cnn.git
cd casting_defect_cnn

```

Install the required packages:

```
pip install -r requirements.txt

```

Make sure the dataset is placed inside the `data/casting_data/` directory with the expected folder structure.

Run the project:

```
python main.py

```

The model will be trained, evaluated, and saved inside the `models/` directory. Training graphs and evaluation plots will be saved inside the `results/` directory.



# Terminal Output :
Dataset downloaded successfully!
Dataset location: /Users/saivenkatr/.cache/kagglehub/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product/versions/2

Dataset structure found!
Train directory: /Users/saivenkatr/.cache/kagglehub/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product/versions/2/casting_data/casting_data/train
Test directory: /Users/saivenkatr/.cache/kagglehub/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product/versions/2/casting_data/casting_data/test

Dataset Statistics

Training directory:
ok_front: 2875
def_front: 3758

Testing directory:
ok_front: 262
def_front: 453
Found 6633 files belonging to 2 classes.
Using 5307 files for training.
Found 6633 files belonging to 2 classes.
Using 1326 files for validation.
Found 715 files belonging to 2 classes.

Class Mapping:
0 = ok_front
1 = def_front

Class mapping: ['ok_front', 'def_front']

Actual validation labels:
def_front (label 1): 1326

Predicted validation labels:
ok_front (label 0): 1326

Confusion Matrix:
[[   0    0]
 [1326    0]]

Probability Statistics:
Min: 0.30864555
Max: 0.36477715
Mean: 0.3384009

First 20 probabilities:
[0.35618818 0.34645945 0.3443079  0.35370526 0.34435213 0.344223
 0.3475521  0.35636538 0.32660663 0.3540105  0.3217201  0.34070614
 0.3141822  0.35878193 0.3429274  0.3470484  0.32977    0.34498087
 0.33401138 0.33825555]
(.venv) (base) saivenkatr@mac casting_defect_cnn 3 % python main.py                                                               

Downloading Kaggle dataset...

Dataset downloaded successfully!
Dataset location: /Users/saivenkatr/.cache/kagglehub/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product/versions/2

Dataset structure found!
Train directory: /Users/saivenkatr/.cache/kagglehub/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product/versions/2/casting_data/casting_data/train
Test directory: /Users/saivenkatr/.cache/kagglehub/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product/versions/2/casting_data/casting_data/test

==================================================
DATASET STATISTICS
==================================================

Training directory:
ok_front: 2875
def_front: 3758

Testing directory:
ok_front: 262
def_front: 453

Total training images: 6633
Training images: 5306
Validation images: 1327

Training class distribution:
ok_front: 2300
def_front: 3006

Validation class distribution:
ok_front: 575
def_front: 752

Total testing images: 715

Testing class distribution:
ok_front: 262
def_front: 453

==================================================
CLASS MAPPING
==================================================
0 = ok_front
1 = def_front
Model: "casting_defect_cnn"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ rescaling (Rescaling)                │ (None, 224, 224, 1)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d (Conv2D)                      │ (None, 224, 224, 32)        │             288 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization                  │ (None, 224, 224, 32)        │             128 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ activation (Activation)              │ (None, 224, 224, 32)        │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 224, 224, 32)        │           9,216 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_1                │ (None, 224, 224, 32)        │             128 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ activation_1 (Activation)            │ (None, 224, 224, 32)        │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 112, 112, 32)        │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_2 (Conv2D)                    │ (None, 112, 112, 64)        │          18,432 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_2                │ (None, 112, 112, 64)        │             256 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ activation_2 (Activation)            │ (None, 112, 112, 64)        │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_3 (Conv2D)                    │ (None, 112, 112, 64)        │          36,864 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_3                │ (None, 112, 112, 64)        │             256 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ activation_3 (Activation)            │ (None, 112, 112, 64)        │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 56, 56, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_4 (Conv2D)                    │ (None, 56, 56, 128)         │          73,728 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_4                │ (None, 56, 56, 128)         │             512 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ activation_4 (Activation)            │ (None, 56, 56, 128)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_5 (Conv2D)                    │ (None, 56, 56, 128)         │         147,456 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_5                │ (None, 56, 56, 128)         │             512 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ activation_5 (Activation)            │ (None, 56, 56, 128)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_2 (MaxPooling2D)       │ (None, 28, 28, 128)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_6 (Conv2D)                    │ (None, 28, 28, 256)         │         294,912 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_6                │ (None, 28, 28, 256)         │           1,024 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ activation_6 (Activation)            │ (None, 28, 28, 256)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_7 (Conv2D)                    │ (None, 28, 28, 256)         │         589,824 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_7                │ (None, 28, 28, 256)         │           1,024 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ activation_7 (Activation)            │ (None, 28, 28, 256)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_3 (MaxPooling2D)       │ (None, 14, 14, 256)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ global_average_pooling2d             │ (None, 256)                 │               0 │
│ (GlobalAveragePooling2D)             │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 256)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 1)                   │             257 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 1,174,817 (4.48 MB)
 Trainable params: 1,172,897 (4.47 MB)
 Non-trainable params: 1,920 (7.50 KB)

Class weights:
{0: 1.1534782608695653, 1: 0.8825681969394544}
/Users/saivenkatr/Downloads/casting_defect_cnn 3/.venv/lib/python3.12/site-packages/keras/src/trainers/epoch_iterator.py:74: UserWarning: `shuffle=True` was passed, but will be ignored since the data `x` was provided as a tf.data.Dataset. The Dataset is expected to already be shuffled (via `.shuffle(buffer_size)`).
  self.data_adapter = data_adapters.get_data_adapter(
Epoch 1/10
166/166 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - accuracy: 0.9014 - auc: 0.9611 - loss: 0.2507 - precision: 0.9345 - recall: 0.8882  
Epoch 1: val_loss improved from None to 0.72113, saving model to models/casting_cnn.keras

Epoch 1: finished saving model to models/casting_cnn.keras
166/166 ━━━━━━━━━━━━━━━━━━━━ 266s 2s/step - accuracy: 0.9014 - auc: 0.9611 - loss: 0.2507 - precision: 0.9345 - recall: 0.8882 - val_accuracy: 0.5667 - val_auc: 0.5000 - val_loss: 0.7211 - val_precision: 0.5667 - val_recall: 1.0000 - learning_rate: 0.0010
Epoch 2/10
166/166 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - accuracy: 0.9572 - auc: 0.9902 - loss: 0.1240 - precision: 0.9737 - recall: 0.9501  
Epoch 2: val_loss improved from 0.72113 to 0.69694, saving model to models/casting_cnn.keras

Epoch 2: finished saving model to models/casting_cnn.keras
166/166 ━━━━━━━━━━━━━━━━━━━━ 264s 2s/step - accuracy: 0.9572 - auc: 0.9902 - loss: 0.1240 - precision: 0.9737 - recall: 0.9501 - val_accuracy: 0.5667 - val_auc: 0.8414 - val_loss: 0.6969 - val_precision: 0.5667 - val_recall: 1.0000 - learning_rate: 0.0010
Epoch 3/10
166/166 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - accuracy: 0.9700 - auc: 0.9941 - loss: 0.0936 - precision: 0.9830 - recall: 0.9637  
Epoch 3: val_loss did not improve from 0.69694
166/166 ━━━━━━━━━━━━━━━━━━━━ 269s 2s/step - accuracy: 0.9700 - auc: 0.9941 - loss: 0.0936 - precision: 0.9830 - recall: 0.9637 - val_accuracy: 0.5667 - val_auc: 0.8389 - val_loss: 1.0554 - val_precision: 0.5667 - val_recall: 1.0000 - learning_rate: 0.0010
Epoch 4/10
166/166 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - accuracy: 0.9747 - auc: 0.9954 - loss: 0.0789 - precision: 0.9848 - recall: 0.9704  
Epoch 4: val_loss did not improve from 0.69694

Epoch 4: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
166/166 ━━━━━━━━━━━━━━━━━━━━ 282s 2s/step - accuracy: 0.9747 - auc: 0.9954 - loss: 0.0789 - precision: 0.9848 - recall: 0.9704 - val_accuracy: 0.5772 - val_auc: 0.7789 - val_loss: 1.6050 - val_precision: 0.5727 - val_recall: 1.0000 - learning_rate: 0.0010
Epoch 5/10
166/166 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - accuracy: 0.9844 - auc: 0.9977 - loss: 0.0515 - precision: 0.9909 - recall: 0.9814  
Epoch 5: val_loss improved from 0.69694 to 0.67627, saving model to models/casting_cnn.keras

Epoch 5: finished saving model to models/casting_cnn.keras
166/166 ━━━━━━━━━━━━━━━━━━━━ 273s 2s/step - accuracy: 0.9844 - auc: 0.9977 - loss: 0.0515 - precision: 0.9909 - recall: 0.9814 - val_accuracy: 0.6466 - val_auc: 0.8081 - val_loss: 0.6763 - val_precision: 0.6411 - val_recall: 0.8551 - learning_rate: 5.0000e-04
Epoch 6/10
166/166 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - accuracy: 0.9891 - auc: 0.9989 - loss: 0.0359 - precision: 0.9963 - recall: 0.9844  
Epoch 6: val_loss did not improve from 0.67627
166/166 ━━━━━━━━━━━━━━━━━━━━ 364s 2s/step - accuracy: 0.9891 - auc: 0.9989 - loss: 0.0359 - precision: 0.9963 - recall: 0.9844 - val_accuracy: 0.6081 - val_auc: 0.9766 - val_loss: 1.0618 - val_precision: 0.5912 - val_recall: 1.0000 - learning_rate: 5.0000e-04
Epoch 7/10
166/166 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.9910 - auc: 0.9989 - loss: 0.0343 - precision: 0.9960 - recall: 0.9880  
Epoch 7: val_loss did not improve from 0.67627

Epoch 7: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.
166/166 ━━━━━━━━━━━━━━━━━━━━ 263s 2s/step - accuracy: 0.9910 - auc: 0.9989 - loss: 0.0343 - precision: 0.9960 - recall: 0.9880 - val_accuracy: 0.6021 - val_auc: 0.8661 - val_loss: 1.7498 - val_precision: 0.5875 - val_recall: 1.0000 - learning_rate: 5.0000e-04
Epoch 8/10
166/166 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.9938 - auc: 0.9996 - loss: 0.0221 - precision: 0.9963 - recall: 0.9927  
Epoch 8: val_loss improved from 0.67627 to 0.02991, saving model to models/casting_cnn.keras

Epoch 8: finished saving model to models/casting_cnn.keras
166/166 ━━━━━━━━━━━━━━━━━━━━ 262s 2s/step - accuracy: 0.9938 - auc: 0.9996 - loss: 0.0221 - precision: 0.9963 - recall: 0.9927 - val_accuracy: 0.9925 - val_auc: 0.9999 - val_loss: 0.0299 - val_precision: 0.9882 - val_recall: 0.9987 - learning_rate: 2.5000e-04
Epoch 9/10
166/166 ━━━━━━━━━━━━━━━━━━━━ 0s 1s/step - accuracy: 0.9940 - auc: 0.9992 - loss: 0.0226 - precision: 0.9963 - recall: 0.9930  
Epoch 9: val_loss improved from 0.02991 to 0.01215, saving model to models/casting_cnn.keras

Epoch 9: finished saving model to models/casting_cnn.keras
166/166 ━━━━━━━━━━━━━━━━━━━━ 262s 2s/step - accuracy: 0.9940 - auc: 0.9992 - loss: 0.0226 - precision: 0.9963 - recall: 0.9930 - val_accuracy: 0.9977 - val_auc: 1.0000 - val_loss: 0.0121 - val_precision: 1.0000 - val_recall: 0.9960 - learning_rate: 2.5000e-04
Epoch 10/10
166/166 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - accuracy: 0.9945 - auc: 0.9992 - loss: 0.0228 - precision: 0.9980 - recall: 0.9923  
Epoch 10: val_loss did not improve from 0.01215
166/166 ━━━━━━━━━━━━━━━━━━━━ 423s 3s/step - accuracy: 0.9945 - auc: 0.9992 - loss: 0.0228 - precision: 0.9980 - recall: 0.9923 - val_accuracy: 0.9962 - val_auc: 0.9999 - val_loss: 0.0298 - val_precision: 1.0000 - val_recall: 0.9934 - learning_rate: 2.5000e-04
Restoring model weights from the end of the best epoch: 9.

Model saved to: models/casting_cnn.keras

Validation threshold tuning
Best threshold: 0.1
Best validation F1: 0.9993346640053227

Test Results
Threshold : 0.1
Accuracy  : 0.9986013986013986
Precision : 0.9977973568281938
Recall    : 1.0
F1-score  : 0.9988974641675854
ROC-AUC   : 1.0

Confusion Matrix
[[261   1]
 [  0 453]]

Classification Report
              precision    recall  f1-score   support

    ok_front       1.00      1.00      1.00       262
   def_front       1.00      1.00      1.00       453

    accuracy                           1.00       715
   macro avg       1.00      1.00      1.00       715
weighted avg       1.00      1.00      1.00       715


Probability Statistics
Min: 1.8924629e-06
Max: 1.0
Mean: 0.6240292
