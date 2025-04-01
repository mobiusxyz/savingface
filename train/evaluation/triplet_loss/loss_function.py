import tensorflow as tf
from tensorflow.keras import backend as K

# closure function to define the triplet loss function
def triplet_loss(emb_dim, alpha=0.2):
    def loss(y_true, y_pred):
        anc, pos, neg = y_pred[:, :emb_dim], y_pred[:, emb_dim:emb_dim*2], y_pred[:, 2*emb_dim:]
        dp = tf.reduce_mean(tf.square(anc - pos), axis=1)
        dn = tf.reduce_mean(tf.square(anc - neg), axis=1)
        return tf.maximum(dp-dn+alpha, 0.)
    return loss