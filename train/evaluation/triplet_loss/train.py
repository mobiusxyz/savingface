import tensorflow as tf
import numpy as np
from loss_function import triplet_loss
from network import tower, siamese_network
from sklearn.model_selection import train_test_split
from data_loader import create_batch
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

emb_dim = 128
batch_size = 32
epochs = 10
#steps_per_epoch = int(60000/batch_size)
steps_per_epoch = int(10000/batch_size)
val_steps_per_epoch = int(10000/batch_size)

print('TensorFlow version:', tf.__version__)

net = siamese_network(tower(embed_dim=emb_dim))
net.summary()
net.compile(loss=triplet_loss(emb_dim=emb_dim, alpha=0.2), optimizer='adam')

#anchors, positives, negatives = create_batch(1000)
# plot_triplet(anchors[0], positives[0], negatives[0])

def data_generator(batch_size, training_ids, emb_dim):
    while True:
        x = create_batch(batch_size, training_ids)
        y = np.zeros((batch_size, 3*emb_dim))
        yield x, y

# split the train and test data

# Load the CSV file
csv_data = pd.read_csv('src/train/evaluation/triplet_loss/training_set.csv')

# Split the data into image filenames and IDs
image_filenames = csv_data['image'].values
ids = csv_data['id'].unique()

# Split the data into training and test sets based on IDs
train_ids, test_ids = train_test_split(ids, test_size=0.2, random_state=45)

# Define the log directory for TensorBoard
log_dir = 'logs/'

# Create the TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

# Define the early stopping callback
early_stopping_callback = EarlyStopping(patience=3)

# Define the model checkpoint callback
checkpoint_filepath = 'models/sfcnn-res50-triplet.weights'
checkpoint_callback = ModelCheckpoint( filepath=checkpoint_filepath, save_weights_only=True, mode='max', save_best_only=True)

# Train your model with the TensorBoard callback
_ = net.fit(
    data_generator(batch_size, train_ids, emb_dim),
    epochs=epochs, 
    steps_per_epoch=steps_per_epoch,
    validation_data=data_generator(batch_size, test_ids, emb_dim),
    validation_steps=val_steps_per_epoch,
    verbose=True,
    callbacks=[tensorboard_callback, early_stopping_callback, checkpoint_callback]
)