import tensorflow as tf
from tensorflow import keras as K

class SoftNcut_loss(K.losses.Loss):
    def __init__(self, K, N):
        super(SoftNcut_loss, self).__init__()
        self.K = K # num class
        self.N = N # image row * col
    
    def call(self, weights, img_seg):
        batch_size = tf.cast(img_seg.shape[0], dtype=tf.int32)
        weights_ = tf.transpose(weights[tf.newaxis, ...], [1, 0, 2, 3]) 
        flat_img_seg= tf.reshape(img_seg, [batch_size, self.N, 1, self.K])
        prob_product = tf.linalg.matmul(tf.transpose(flat_img_seg, [0, 3, 1, 2]), tf.transpose(flat_img_seg, [0, 3, 2, 1]))
        
        assoc_nomi = tf.math.reduce_sum(weights_ *  prob_product)
        assoc_denomi = tf.math.reduce_sum(tf.transpose(flat_img_seg, [0, 3, 1, 2]) * weights_)
        
        softncut_loss = self.K - assoc_nomi / assoc_denomi
        
        return softncut_loss

class Recon_loss(K.losses.Loss):
    def __init__(self):
        super(Recon_loss, self).__init__()
        
    def call(self, truth, recon):
        recon_loss = tf.math.reduce_sum(K.losses.MSE(truth, recon))
        
        return recon_loss