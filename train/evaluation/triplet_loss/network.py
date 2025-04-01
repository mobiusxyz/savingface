import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, BatchNormalization, Dense, Dropout 
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50, ResNet50V2

# resnet_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
resnet_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# make the resnet model non-trainable
trainable = False
for layer in resnet_model.layers:
    if layer.name == "conv5_block1_out":
        trainable = True
    layer.trainable = trainable
    

def tower(embed_dim=128):
    # build a sequential model with base_model (resnet50) and additional layers
    return tf.keras.models.Sequential([
        resnet_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dropout(0.3),
        Dense(embed_dim, activation='sigmoid')
    ])

# tower(embed_dim=128).summary()

# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  resnet50v2 (Functional)     (None, 7, 7, 2048)        23564800  
#                                                                  
#  global_average_pooling2d (  (None, 2048)              0         
#  GlobalAveragePooling2D)                                         
#                                                                  
#  batch_normalization (Batch  (None, 2048)              8192      
#  Normalization)                                                  
#                                                                  
#  dropout (Dropout)           (None, 2048)              0         
#                                                                  
#  dense (Dense)               (None, 128)               262272    
#                                                                  
# =================================================================
# Total params: 23835264 (90.92 MB)
# Trainable params: 23785728 (90.74 MB)
# Non-trainable params: 49536 (193.50 KB)

def tower_s(embed_dim=64):
    return tf.keras.models.Sequential([
        Dense(64, activation='relu',input_shape=(224, 224, 3)),
        Dense(embed_dim, activation='sigmoid')
    ]) 

## tower_s(embed_dim=64).summary()

# Model: "sequential_1"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  dense_1 (Dense)             (None, 64)                50240     
#                                                                  
#  dense_2 (Dense)             (None, 64)                4160      
#                                                                  
# =================================================================
# Total params: 54400 (212.50 KB)
# Trainable params: 54400 (212.50 KB)
# Non-trainable params: 0 (0.00 Byte)

def siamese_network(embedding_model=tower(embed_dim=128)):
    
    input_anchor = Input(shape=(224, 224, 3), name='anchor')
    input_positive = Input(shape=(224, 224, 3), name='positive')
    input_negative = Input(shape=(224, 224, 3), name='negative')

    embedding_anchor = embedding_model(input_anchor)
    embedding_positive = embedding_model(input_positive)
    embedding_negative = embedding_model(input_negative)

    output = tf.keras.layers.concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1) # column wise that's why
    net = Model([input_anchor, input_positive, input_negative], output) 
    return net
    
# siamese_network(tower(embed_dim=128)).summary() 

# Model: "model"
# __________________________________________________________________________________________________
#  Layer (type)                Output Shape                 Param #   Connected to                  
# ==================================================================================================
#  anchor (InputLayer)         [(None, 224, 224, 3)]        0         []                            
#                                                                                                   
#  positive (InputLayer)       [(None, 224, 224, 3)]        0         []                            
#                                                                                                   
#  negative (InputLayer)       [(None, 224, 224, 3)]        0         []                            
#                                                                                                   
#  sequential_2 (Sequential)   (None, 128)                  2383526   ['anchor[0][0]',              
#                                                           4          'positive[0][0]',            
#                                                                      'negative[0][0]']            
#                                                                                                   
#  concatenate (Concatenate)   (None, 384)                  0         ['sequential_2[0][0]',        
#                                                                      'sequential_2[1][0]',        
#                                                                      'sequential_2[2][0]']        
#                                                                                                   
# ==================================================================================================
# Total params: 23835264 (90.92 MB)
# Trainable params: 23785728 (90.74 MB)
# Non-trainable params: 49536 (193.50 KB)
# __________________________________________________________________________________________________