import numpy as np
import pandas as pd
import random
import cv2
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt

def augment_image(image):
    seq = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.Affine(rotate=(-10, 10))),  # Rotate image by -10 to 10 degrees
        iaa.Sometimes(0.5, iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)})),  # Scale/zoom image by 0.8 to 1.2 times
        iaa.Sometimes(0.5, iaa.Affine(shear=(-10, 10)))  # Apply shear transformation by -10 to 10 degrees
    ], random_order=True)
    augmented_image = seq.augment_image(image)
    return augmented_image

def create_batch(batch_size, unique_ids, image_directory='data/faces/'):

    # create an empty non-np array with the shape of the batch size
    anchors = np.empty((batch_size, 224, 224, 3))
    positives = np.empty((batch_size, 224, 224, 3))
    negatives = np.empty((batch_size, 224, 224, 3))
    
    data = pd.read_csv('train/evaluation/triplet_loss/training_set.csv')
    
    triplet_indices = set()
    
    while len(triplet_indices) < batch_size:
        index = random.randint(0, len(data)-1)
        anchor_id = data.loc[index, 'id']
        
        # Get anchor image
        anchor_filename = data.loc[index, 'image']
        anchor_image = cv2.imread(image_directory + anchor_filename)
        #anchor_image = cv2.resize(anchor_image, (224, 224))
        
        # Get positive image from the same ID
        positive_indices = data[data['id'] == anchor_id].index.values
        positive_index = random.choice(positive_indices)
        positive_filename = data.loc[positive_index, 'image']
        positive_image = cv2.imread(image_directory + positive_filename)
        #positive_image = cv2.resize(positive_image, (224, 224))
        
        # Get negative image from a different ID
        negative_id = random.choice(unique_ids[unique_ids != anchor_id])
        negative_indices = data[data['id'] == negative_id].index.values
        negative_index = random.choice(negative_indices)
        negative_filename = data.loc[negative_index, 'image']
        negative_image = cv2.imread(image_directory + negative_filename)
        #negative_image = cv2.resize(negative_image, (224, 224)
        
        triplet = (index, positive_index, negative_index)
        
        if triplet not in triplet_indices:
            # print("Creating new triplet... id {} {} {}".format(anchor_id, data.loc[positive_index, 'id'], data.loc[negative_index, 'id']))
            triplet_indices.add(triplet)
            anchors[len(triplet_indices) - 1] = anchor_image
            positives[len(triplet_indices) - 1] = positive_image
            negatives[len(triplet_indices) - 1] = negative_image
        else:
            # print("Triplet already exists, augmenting...")
            # Apply augmentation to the anchor, positive, and negative images
            anchor_image = augment_image(anchor_image)
            positive_image = augment_image(positive_image)
            negative_image = augment_image(negative_image)
            anchors[len(triplet_indices) - 1] = anchor_image
            positives[len(triplet_indices) - 1] = positive_image
            negatives[len(triplet_indices) - 1] = negative_image
    
    # make sure all value in np array is integer
    anchors = anchors.astype(np.uint8)
    positives = positives.astype(np.uint8)
    negatives= negatives.astype(np.uint8)
    return anchors, positives, negatives

def plot_triplet(anchor, positive, negative):
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(anchor)
    axes[0].set_title('Anchor')
    axes[0].axis('off')
    axes[1].imshow(positive)
    axes[1].set_title('Positive')
    axes[1].axis('off')
    axes[2].imshow(negative)
    axes[2].set_title('Negative')
    axes[2].axis('off')
    plt.show()