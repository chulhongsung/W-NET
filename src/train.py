import tensorflow as tf
from tensorflow import keras as K
import numpy as np

from loss import SoftNcut_loss, Recon_loss
from wnet import Wnet

weights = np.load("ncut_weights_stl10_30.npy")
train_img = np.load("train_stl10_30.npy")

BATCH_SIZE = 5
stl10_30 = tf.data.Dataset.from_tensor_slices((np.array(weights, dtype=np.float32), train_img))
batch_stl10_30 = stl10_30.batch(BATCH_SIZE)

optimizer = K.optimizers.Adam(learning_rate=0.01)
train_loss = K.metrics.Mean(name='train_loss')

@tf.function
def train_step(model, weights, image):
    with tf.GradientTape(persistent=True) as tape:
        img_seg, recon = model(image)
        sn_loss = softncut_loss(weights, img_seg)
        rc_loss = recon_loss(image, recon)
        loss = sn_loss + rc_loss
    grad1 = tape.gradient(sn_loss, model.unet_encoder.trainable_weights)
    grad2 = tape.gradient(rc_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grad1, model.unet_encoder.trainable_weights))
    optimizer.apply_gradients(zip(grad2, model.trainable_weights))
    train_loss(loss)

wnet = Wnet(10)    

EPOCHS = 100

for epoch in range(EPOCHS):
    for batch_weights, batch_image in iter(batch_stl10_30):
        train_step(wnet, batch_weights, batch_image)
    template = 'EPOCH: {}, Train Loss: {}'
    print(template.format(epoch+1, train_loss.result()))