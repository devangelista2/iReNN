import numpy as np
import matplotlib.pyplot as plt

from IPPy.nn import models
from IPPy import metrics

import tensorflow as tf
from tensorflow import keras as ks

# Disable Tf Warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Set the data path
train_path = './data/iReNN_train.npy'
test_path = './data/iReNN_test.npy'

# Load train data
test_data = np.load(test_path)
N_train, H, M, N = test_data.shape

# Print shapes
print(test_data.shape)

# Test1: Predict Test Set iteration by iteration
test1 = False
if test1:
    for i in range(1, H-1):
        # Load data
        x_test = test_data[:, i, :, :]
        y_test = test_data[:, i+1, :, :]

        # Add channel dimension
        x_test = np.expand_dims(x_test, -1)
        y_test = np.expand_dims(y_test, -1)

        # Load the model
        model = ks.models.load_model(f"./model_weights/iReNN_{i}to{i+1}.h5", custom_objects={'SSIM': metrics.SSIM})

        # Predict
        y_pred = model.predict(x_test, verbose=0)

        # Compute the mean SSIM
        ssim_vec = np.zeros((len(x_test), ))
        for idx in range(len(x_test)):
            ssim_vec[idx] = metrics.SSIM(y_test[idx], y_pred[idx])
        
        print(f"Mean SSIM for iteration {i}: {np.mean(ssim_vec)}, Std.: {np.std(ssim_vec)}.")

# Test2: Predict Test Set using the last prediction as input
test2 = True
if test2:
    # Load data
    x_test = test_data[:, 1, :, :]
    y_test = test_data[:, 2, :, :]

    # Add channel dimension
    x_test = np.expand_dims(x_test, -1)
    y_test = np.expand_dims(y_test, -1)
    for i in range(1, H-1):
        # Load the model
        model = ks.models.load_model(f"./model_weights/iReNN_{i}to{i+1}.h5", custom_objects={'SSIM': metrics.SSIM})

        # Predict
        x_test = model.predict(x_test, verbose=0)

        # Compute the mean SSIM
        ssim_vec = np.zeros((len(x_test), ))
        for idx in range(len(x_test)):
            ssim_vec[idx] = metrics.SSIM(y_test[idx], x_test[idx])
        
        print(f"Mean SSIM for iteration {i}: {np.mean(ssim_vec)}, Std.: {np.std(ssim_vec)}.")
