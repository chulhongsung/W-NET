#%%
import tensorflow as tf
from tensorflow import keras as K
#%%
class Conv_module(K.layers.Layer):
    def __init__(self, out_channel, kernel_size = 3, strides=(1,1)):
        super(Conv_module,  self).__init__()
        self.n_layer = len(out_channel)
        self.conv_layers = [K.layers.Conv2D(filters=i, kernel_size=kernel_size, strides=strides, padding='same', activation='relu') for i in out_channel]
        self.batch_norm = [K.layers.BatchNormalization() for i in range(self.n_layer)]
    
    def call(self, img):
        for i in range(self.n_layer):
            img = self.conv_layers[i](img)
            img = self.batch_norm[i](img)
        return img
#%%
class Dsconv_module(K.layers.Layer):
    def __init__(self, out_channel, kernel_size=3, strides=(1,1)):
        super(Dsconv_module, self).__init__()
        self.n_layer = len(out_channel)
        self.dsconv_layers = [K.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same', depth_multiplier=i, activation='relu') for i in out_channel]
        self.batch_norm = [K.layers.BatchNormalization() for i in range(self.n_layer)]
        
    def call(self, img):
        for i in range(self.n_layer):
            img = self.dsconv_layers[i](img)
            img = self.batch_norm[i](img)
        return img
#%%
class Upconv_module(K.layers.Layer):
    def __init__(self, out_channel, kernel_size=3, strides=(1,1)):
        super(Upconv_module, self).__init__()
        self.n_layer = 2
        self.conv_layer = K.layers.Conv2D(filters = out_channel, kernel_size=kernel_size, strides=strides, padding='same', activation='relu')
        self.dsconv_layer = K.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same', depth_multiplier=1, activation='relu')
        self.batch_norm = [K.layers.BatchNormalization() for i in range(self.n_layer)]
        
    def call(self, img):
        half_ch_img = self.conv_layer(img)
        half_ch_img = self.batch_norm[0](half_ch_img)
        dsconv_img = self.dsconv_layer(half_ch_img)
        dsconv_img = self.batch_norm[1](dsconv_img)
        return dsconv_img
#%%
class Unet_enc(K.layers.Layer):
    def __init__(self, num_class, conv1_out_channel=[64, 64], conv2_out_channel=[64, 64]):
        super(Unet_enc, self).__init__()
        self.conv_module = Conv_module(conv1_out_channel)
        self.down_dsconv_module = [Dsconv_module([2,1]) for i in range(4)]
        
        self.deconv = [K.layers.Conv2DTranspose(filters = i, kernel_size=2, strides=(2,2), padding='valid') for i in [512, 256, 128, 64]]
        self.up_conv_module = [Upconv_module(i) for i in [512, 256, 128, 64]]
        
        self.max_pool = K.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='valid') # img size halved
        
        self.final_conv_module = Conv_module(conv2_out_channel)
        self.pixelwise_conv = K.layers.Conv2D(num_class, kernel_size=1, strides=(1,1), padding='same', activation='relu')
        self.softmax = K.layers.Softmax(axis=-1)
        
    def call(self, img):
        # down-sampling
        module1 = self.conv_module(img) # shape: (448, 384, 64)
        pooled_module1 = self.max_pool(module1) # shape: (224, 192, 64)
        module2 = self.down_dsconv_module[0](pooled_module1) # shape: (224, 192, 128)
        pooled_module2 = self.max_pool(module2) # shape: (112, 96, 128)
        module3 = self.down_dsconv_module[1](pooled_module2) # shape: (112, 96, 256)
        pooled_module3 = self.max_pool(module3) # shape: (56, 48, 256)
        module4 = self.down_dsconv_module[2](pooled_module3) # shape: (56, 48, 512)
        pooled_module4 = self.max_pool(module4) # shape: (28, 24, 512)
        module5 = self.down_dsconv_module[3](pooled_module4) # bottle neck point shape: (28, 24, 1024)
        
        # up-sampling
        deconv_module5 = self.deconv[0](module5) # shape: (56, 48, 512)
        module6 = self.up_conv_module[0](tf.keras.layers.Concatenate()([module4, deconv_module5])) # shape: (56, 48, 512)
        deconv_module6 = self.deconv[1](module6)
        module7 = self.up_conv_module[1](tf.keras.layers.Concatenate()([module3, deconv_module6]))
        deconv_module7 = self.deconv[2](module7)
        module8 = self.up_conv_module[2](tf.keras.layers.Concatenate()([module2, deconv_module7]))
        deconv_module8 = self.deconv[3](module8)
        module9 = self.pixelwise_conv(self.final_conv_module(tf.keras.layers.Concatenate()([module1, deconv_module8])))
        softmax = self.softmax(module9) # shape: (448, 384, num_class=18)
        
        return softmax