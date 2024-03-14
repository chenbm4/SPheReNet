import tensorflow as tf

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, num_outputs, kernel_size=4, stride=1, activation_fn=tf.nn.relu, l2_value=0.0002, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.num_outputs = num_outputs
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation_fn = activation_fn

        l2_regularizer = tf.keras.regularizers.l2(l2_value)

        self.conv1 = tf.keras.layers.Conv2D(num_outputs // 2, 1, strides=1, padding='SAME', kernel_regularizer=l2_regularizer)
        self.conv2 = tf.keras.layers.Conv2D(num_outputs // 2, kernel_size, strides=stride, padding='SAME', kernel_regularizer=l2_regularizer)
        self.conv3 = tf.keras.layers.Conv2D(num_outputs, 1, strides=1, padding='SAME', use_bias=False, kernel_regularizer=l2_regularizer)
        self.batch_norm = tf.keras.layers.BatchNormalization()

        self.shortcut_conv = tf.keras.layers.Conv2D(num_outputs, 1, strides=stride, 
                                                    padding='SAME', use_bias=False)

    def call(self, x, training=False):
        shortcut = self.shortcut_conv(x) if self.stride != 1 or x.shape[-1] != self.num_outputs else x
        x = self.conv1(x)
        x = self.activation_fn(self.batch_norm(x, training=training))
        x = self.conv2(x)
        x = self.activation_fn(self.batch_norm(x, training=training))
        x = self.conv3(x)
        x = self.batch_norm(x, training=training)  # No activation after this layer
        x = x + shortcut
        return self.activation_fn(x)

class ResFcn256(tf.keras.Model):
    def __init__(self, resolution_inp=256, resolution_op=512, channel=3, l2_value=0.0002):
        super(ResFcn256, self).__init__()
        size = 16
        l2_regularizer = tf.keras.regularizers.l2(l2_value)
        self.initial_conv = tf.keras.layers.Conv2D(size, kernel_size=4, strides=1, padding='SAME', kernel_regularizer=l2_regularizer)
        self.initial_batch_norm = tf.keras.layers.BatchNormalization()

        self.res_blocks = [
            ResBlock(size * 2, kernel_size=4, stride=2, l2_value=l2_value),
            ResBlock(size * 2, kernel_size=4, stride=1, l2_value=l2_value),
            ResBlock(size * 4, kernel_size=4, stride=2, l2_value=l2_value),
            ResBlock(size * 4, kernel_size=4, stride=1, l2_value=l2_value),
            ResBlock(size * 8, kernel_size=4, stride=2, l2_value=l2_value),
            ResBlock(size * 8, kernel_size=4, stride=1, l2_value=l2_value),
            ResBlock(size * 16, kernel_size=4, stride=2, l2_value=l2_value),
            ResBlock(size * 16, kernel_size=4, stride=1, l2_value=l2_value),
            ResBlock(size * 32, kernel_size=4, stride=2, l2_value=l2_value),
            ResBlock(size * 32, kernel_size=4, stride=1, l2_value=l2_value)
        ]

        self.upconvs = [
            tf.keras.layers.Conv2DTranspose(size * 32, 4, strides=1, padding='SAME', kernel_regularizer=l2_regularizer),
            tf.keras.layers.Conv2DTranspose(size * 16, 4, strides=2, padding='SAME', kernel_regularizer=l2_regularizer),
            tf.keras.layers.Conv2DTranspose(size * 16, 4, strides=1, padding='SAME', kernel_regularizer=l2_regularizer),
            tf.keras.layers.Conv2DTranspose(size * 16, 4, strides=1, padding='SAME', kernel_regularizer=l2_regularizer),
            tf.keras.layers.Conv2DTranspose(size * 8, 4, strides=2, padding='SAME', kernel_regularizer=l2_regularizer),
            tf.keras.layers.Conv2DTranspose(size * 8, 4, strides=1, padding='SAME', kernel_regularizer=l2_regularizer),
            tf.keras.layers.Conv2DTranspose(size * 8, 4, strides=1, padding='SAME', kernel_regularizer=l2_regularizer),
            tf.keras.layers.Conv2DTranspose(size * 4, 4, strides=2, padding='SAME', kernel_regularizer=l2_regularizer),
            tf.keras.layers.Conv2DTranspose(size * 4, 4, strides=1, padding='SAME', kernel_regularizer=l2_regularizer),
            tf.keras.layers.Conv2DTranspose(size * 4, 4, strides=1, padding='SAME', kernel_regularizer=l2_regularizer),

            tf.keras.layers.Conv2DTranspose(size * 2, 4, strides=2, padding='SAME', kernel_regularizer=l2_regularizer),
            tf.keras.layers.Conv2DTranspose(size * 2, 4, strides=1, padding='SAME', kernel_regularizer=l2_regularizer),
            tf.keras.layers.Conv2DTranspose(size, 4, strides=2, padding='SAME', kernel_regularizer=l2_regularizer),
            tf.keras.layers.Conv2DTranspose(size, 4, strides=1, padding='SAME', kernel_regularizer=l2_regularizer),

            tf.keras.layers.Conv2DTranspose(1, 4, strides=1, padding='SAME', kernel_regularizer=l2_regularizer),
            tf.keras.layers.Conv2DTranspose(1, 4, strides=1, padding='SAME', kernel_regularizer=l2_regularizer),
            tf.keras.layers.Conv2DTranspose(1, 4, strides=2, padding='SAME', kernel_regularizer=l2_regularizer, activation='sigmoid')
        ]

        self.upconv_batch_norms = [tf.keras.layers.BatchNormalization() for _ in self.upconvs[:-1]]

    def call(self, x, training=False):
        x = self.initial_conv(x)
        x = self.initial_batch_norm(x, training=training)
        for res_block in self.res_blocks:
            x = res_block(x, training=training)
        for upconv, batch_norm in zip(self.upconvs[:-1], self.upconv_batch_norms):
            x = upconv(x)
            x = batch_norm(x, training=training)
        x = self.upconvs[-1](x)
        return x
    