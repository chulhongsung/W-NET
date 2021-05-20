import tensorflow as tf
from tensorflow import keras as K

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
        module1 = self.conv_module(img) # shape: (batch_size, 96, 96, 64)
        pooled_module1 = self.max_pool(module1) # shape: (batch_size, 48, 48, 64)
        module2 = self.down_dsconv_module[0](pooled_module1) # shape: (batch_size, 48, 48, 128)
        pooled_module2 = self.max_pool(module2) # shape: (batch_size, 24, 24, 128)
        module3 = self.down_dsconv_module[1](pooled_module2) # shape: (batch_size, 24, 24, 256)
        pooled_module3 = self.max_pool(module3) # shape: (batch_size, 12, 12, 256)
        module4 = self.down_dsconv_module[2](pooled_module3) # shape: (batch_size, 12, 12, 512)
        pooled_module4 = self.max_pool(module4) # shape: (batch_size, 6, 6, 512)
        module5 = self.down_dsconv_module[3](pooled_module4) # bottle neck point shape: (batch_size, 6, 6, 1024)
        
        # up-sampling
        deconv_module5 = self.deconv[0](module5) # shape: (batch_size, 6, 6, 512)
        module6 = self.up_conv_module[0](tf.keras.layers.Concatenate()([module4, deconv_module5])) # shape: (batch_size, 12, 12, 512)
        deconv_module6 = self.deconv[1](module6) # shape: (batch_size, 12, 12, 256)
        module7 = self.up_conv_module[1](tf.keras.layers.Concatenate()([module3, deconv_module6])) # shape: (batch_size, 24, 24, 256)
        deconv_module7 = self.deconv[2](module7) # shape: (batch_size, 48, 48, 128)
        module8 = self.up_conv_module[2](tf.keras.layers.Concatenate()([module2, deconv_module7])) # shape: (batch_size, 96, 96, 128)
        deconv_module8 = self.deconv[3](module8) # shape: (batch_size, 96, 96, 64)
        module9 = self.pixelwise_conv(self.final_conv_module(tf.keras.layers.Concatenate()([module1, deconv_module8]))) # shape: (batch_size, 96, 96, num_class)
        softmax = self.softmax(module9) # shape: (batch_size, 96, 96, num_class)
        
        return softmax

class Unet_dec(K.layers.Layer):
    def __init__(self, conv1_out_channel=[64, 64], conv2_out_channel=[64, 64]):
        super(Unet_dec, self).__init__()
        self.conv_module = Conv_module(conv1_out_channel)
        self.down_dsconv_module = [Dsconv_module([2,1]) for i in range(4)]
        
        self.deconv = [K.layers.Conv2DTranspose(filters = i, kernel_size=2, strides=(2,2), padding='valid') for i in [512, 256, 128, 64]]
        self.up_conv_module = [Upconv_module(i) for i in [512, 256, 128, 64]]
        
        self.max_pool = K.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='valid') # img size halved
        
        self.final_conv_module = Conv_module(conv2_out_channel)
        self.pixelwise_conv = K.layers.Conv2D(3, kernel_size=1, strides=(1,1), padding='same', activation='relu')
        
    def call(self, img):
        # down-sampling
        module10 = self.conv_module(img) # shape: (batch_size, 448, 384, 64)
        pooled_module10 = self.max_pool(module10) # shape: (batch_size, 224, 192, 64)
        module11 = self.down_dsconv_module[0](pooled_module10) # shape: (batch_size, 224, 192, 128)
        pooled_module11 = self.max_pool(module11) # shape: (batch_size, 112, 96, 128)
        module12 = self.down_dsconv_module[1](pooled_module11) # shape: (batch_size, 112, 96, 256)
        pooled_module12 = self.max_pool(module12) # shape: (batch_size, 56, 48, 256)
        module13 = self.down_dsconv_module[2](pooled_module12) # shape: (batch_size, 56, 48, 512)
        pooled_module13 = self.max_pool(module13) # shape: (batch_size, 28, 24, 512)
        module14 = self.down_dsconv_module[3](pooled_module13) # bottle neck point shape: (batch_size, 28, 24, 1024)
        
        # up-sampling
        deconv_module14 = self.deconv[0](module14) # shape: (batch_size, 56, 48, 512)
        module15 = self.up_conv_module[0](tf.keras.layers.Concatenate()([module13, deconv_module14])) # shape: (batch_size, 56, 48, 512)
        deconv_module15 = self.deconv[1](module15)
        module16 = self.up_conv_module[1](tf.keras.layers.Concatenate()([module12, deconv_module15]))
        deconv_module16 = self.deconv[2](module16)
        module17 = self.up_conv_module[2](tf.keras.layers.Concatenate()([module11, deconv_module16]))
        deconv_module17 = self.deconv[3](module17)
        module18 = self.pixelwise_conv(self.final_conv_module(tf.keras.layers.Concatenate()([module10, deconv_module17]))) # recon (batch_size, 448, 384, 3)
        
        return module18

class Wnet(K.models.Model):
    def __init__(self, num_class, conv1_out_channel=[64, 64], conv2_out_channel=[64, 64]):
        super(Wnet, self).__init__()
        self.unet_encoder = Unet_enc(num_class, conv1_out_channel, conv2_out_channel)
        self.unet_decoder = Unet_dec(conv1_out_channel, conv2_out_channel)
        
    def call(self, img):
        softmax = self.unet_encoder(img) # shape: (batch_size, 448, 384, num_class)
        recon_img = self.unet_decoder(softmax) # shape: (batch_size, 448, 384, 3)
        
        return softmax, recon_img
