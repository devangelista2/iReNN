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
true_train_path = './data/COULE_train'
true_test_path = './data/COULE_test'
train_path = './data/iReNN_train.npy'
test_path = './data/iReNN_test.npy'

# Load train data
test_data = np.load(test_path)
N_train, H, M, N = test_data.shape

# Print shapes
print(test_data.shape)


# Test 0: Use the iCP-TV solutions
test0 = False
if test0:
    # Initialization
    rel_err_total = np.zeros((H-2, ))
    ssim_total = np.zeros((H-2, ))
    for i in range(1, H-1):
        y_test = test_data[:, i+1, :, :]

        # Add channel dimension
        y_test = np.expand_dims(y_test, -1)

        # Compute the mean RE and SSIM
        rel_err_vec = np.zeros((len(y_test), ))
        ssim_vec = np.zeros((len(y_test), ))
        for idx in range(len(y_test)):
            # Load true data
            y_true = np.load(f"{true_test_path}/{idx}.npy")

            # Compute and append the errors
            rel_err_vec[idx] = metrics.rel_err(y_test[idx].flatten(), y_true.flatten())
            ssim_vec[idx] = metrics.SSIM(y_true, y_test[idx])

        print(f"Iteration {i}. RE -> Mean: {np.mean(rel_err_vec):0.4f}, Std.: {np.std(rel_err_vec):0.4f}. SSIM -> Mean: {np.mean(ssim_vec):0.4f}, Std.: {np.std(ssim_vec):0.4f}.")
        
        # Save the results
        rel_err_total[i-1] = np.mean(rel_err_vec)
        ssim_total[i-1] = np.mean(ssim_vec)

    # Save
    np.save('./results/test0/rel_err.npy', rel_err_total)
    np.save('./results/test0/ssim.npy', ssim_total)

# Test1: Predict Test Set iteration by iteration
test1 = False
if test1:
    # Initialization
    rel_err_total = np.zeros((H-2, ))
    ssim_total = np.zeros((H-2, ))
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

        # Compute the mean RE and SSIM
        rel_err_vec = np.zeros((len(x_test), ))
        ssim_vec = np.zeros((len(x_test), ))
        for idx in range(len(x_test)):
            # Load true data
            y_true = np.load(f"{true_test_path}/{idx}.npy")

            # Compute and append the errors
            rel_err_vec[idx] = metrics.rel_err(y_pred[idx].flatten(), y_true.flatten())
            ssim_vec[idx] = metrics.SSIM(y_true, y_pred[idx])
        
        # Save the results
        rel_err_total[i-1] = np.mean(rel_err_vec)
        ssim_total[i-1] = np.mean(ssim_vec)
        
        print(f"Iteration {i}. RE -> Mean: {np.mean(rel_err_vec):0.4f}, Std.: {np.std(rel_err_vec):0.4f}. SSIM -> Mean: {np.mean(ssim_vec):0.4f}, Std.: {np.std(ssim_vec):0.4f}.")

    # Save
    np.save('./results/test1/rel_err.npy', rel_err_total)
    np.save('./results/test1/ssim.npy', ssim_total)

# Test2: Predict Test Set using the last prediction as input
test2 = False
if test2:
    # Load data
    x_test = test_data[:, 1, :, :]
    y_test = test_data[:, 2, :, :]

    # Add channel dimension
    x_test = np.expand_dims(x_test, -1)
    y_test = np.expand_dims(y_test, -1)

    # Initialization
    rel_err_total = np.zeros((H-2, ))
    ssim_total = np.zeros((H-2, ))
    for i in range(1, H-1):
        # Load the model
        model = ks.models.load_model(f"./model_weights/iReNN_{i}to{i+1}.h5", custom_objects={'SSIM': metrics.SSIM})

        # Predict
        x_test = model.predict(x_test, verbose=0)

        # Compute the mean SSIM
        rel_err_vec = np.zeros((len(x_test), ))
        ssim_vec = np.zeros((len(x_test), ))
        for idx in range(len(x_test)):
            # Load true data
            y_true = np.load(f"{true_test_path}/{idx}.npy")
            
            rel_err_vec[idx] = metrics.rel_err(x_test[idx].flatten(), y_true.flatten())
            ssim_vec[idx] = metrics.SSIM(y_true, x_test[idx])

        # Save the results
        rel_err_total[i-1] = np.mean(rel_err_vec)
        ssim_total[i-1] = np.mean(ssim_vec)
        
        print(f"Iteration {i}. RE -> Mean: {np.mean(rel_err_vec):0.4f}, Std.: {np.std(rel_err_vec):0.4f}. SSIM -> Mean: {np.mean(ssim_vec):0.4f}, Std.: {np.std(ssim_vec):0.4f}.")
    
    # Save
    np.save('./results/test2/rel_err.npy', rel_err_total)
    np.save('./results/test2/ssim.npy', ssim_total)