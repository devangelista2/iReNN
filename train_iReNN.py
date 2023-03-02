import numpy as np
import matplotlib.pyplot as plt

from IPPy.nn import models
from IPPy import metrics

import tensorflow as tf
from tensorflow import keras as ks

# Set the data path
train_path = './data/iReNN_train.npy'
test_path = './data/iReNN_test.npy'

# Load train data
train_data = np.load(train_path)
N_train, H, M, N = train_data.shape

# Print shapes
print(train_data.shape)

for i in range(1, H-1):
    # Load data
    x_train = train_data[:, i, :, :]
    y_train = train_data[:, i+1, :, :]

    # Add channel dimension
    x_train = np.expand_dims(x_train, -1)
    y_train = np.expand_dims(y_train, -1)

    # Define the model and compile it
    # Number of epochs
    n_epochs = 50

    # Build model and compile it
    model = models.get_UNet(input_shape = (M, N, 1), n_scales = 2, conv_per_scale = 2, final_relu=True, skip_connection=True)

    # Define the Optimizer
    learning_rate = 1e-3

    model.compile(optimizer=ks.optimizers.Adam(learning_rate=learning_rate),
                loss='mse',
                metrics=[metrics.SSIM, 'mse'])
    
    # Train the model
    hist = model.fit(x_train, y_train, batch_size=16, epochs=n_epochs, validation_split=0.1)
    model.save(f"./model_weights/iReNN_{i}to{i+1}.h5")
    print(f"Training of {i}to{i+1} model -> Finished.")