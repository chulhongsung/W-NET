import tensorflow as tf
from tensorflow import keras as K
import numpy as np

from loss import SoftNcut_loss, Recon_loss
from wnet import Wnet
from crf import crf_batch_fit_predict

weights = np.load("ncut_weights_stl10_30.npy")
train_img = np.load("train_stl10_30.npy")

BATCH_SIZE = 5
NUM_CLASS = 8

stl10_30 = tf.data.Dataset.from_tensor_slices((np.array(weights, dtype=np.float32), train_img))
batch_stl10_30 = stl10_30.batch(BATCH_SIZE)

softncut_loss = SoftNcut_loss(NUM_CLASS, 9216)
recon_loss = Recon_loss()
wnet = Wnet(NUM_CLASS)

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

EPOCHS = 100

for epoch in range(EPOCHS):
    for batch_weights, batch_image in iter(batch_stl10_30):
        train_step(wnet, batch_weights, batch_image)
    template = 'EPOCH: {}, Train Loss: {}'
    print(template.format(epoch+1, train_loss.result()))

wnet.save_weights('./model/wnet_' + datetime.today().strftime("%Y%m%d%H%M%S"))
# wnet2 = Wnet(10)
# wnet2.load_weights('./model/wnet')

img_seg, recon = wnet(train_img)

NITER = 150

softmax = tf.transpose(img_seg, [0, 3, 1, 2])
train_img_crf = np.transpose(train_img, [0, 3, 1, 2])

crf_img = crf_batch_fit_predict(softmax, train_img_crf, NITER)

crf_img = np.transpose(crf_img, [0, 2, 3, 1])

fig = plt.figure()

rows = 2
cols = 2

ax1 = fig.add_subplot(rows, cols, 1)
ax1.imshow(train_img[0].numpy())
ax1.set_title('Image')
ax1.axis("off")

ax2 = fig.add_subplot(rows, cols, 2)
ax2.imshow(tf.clip_by_value(tf.squeeze(recon), 0.0, 1.0).numpy())
ax2.set_title('Reconstruction')
ax2.axis("off")

ax3 = fig.add_subplot(rows, cols, 3)
ax3.imshow(tf.transpose(tf.math.argmax(img_seg[0], axis=-1, output_type=tf.dtypes.int64), [1, 2, 0]).numpy())
ax3.set_title("Segmentation")
ax3.axis("off")

ax4 = fig.add_subplot(rows, cols, 4)
ax4.imshow(np.squeeze(np.argmax(crf_img[0], axis=-1)))
ax4.set_title("CRF")
ax4.axis("off")

plt.show()
plt.close()