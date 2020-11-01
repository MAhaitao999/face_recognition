from keras import layers
from keras.layers import Activation, Convolution2D, Dropout, Conv2D, Dense
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras.regularizers import l2

import numpy as np
import tensorflow as tf

def AlexNet(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(filters=96, kernel_size=(11, 11),
                     strides=4, padding="same",
                     activation="relu", input_shape=input_shape,
                     kernel_initializer="he_normal"))
    
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                           padding="same", data_format=None))

    model.add(Conv2D(filters=256, kernel_size=(5, 5),
                     strides=1, padding="same",
                     activation="relu", kernel_initializer="he_normal"))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                           padding="same", data_format=None))

    model.add(Conv2D(filters=384, kernel_size=(3, 3),
                     strides=1, padding="same",
                     activation="relu", kernel_initializer="he_normal"))

    model.add(Conv2D(filters=384, kernel_size=(3, 3),
                     strides=1, padding="same",
                     activation="relu", kernel_initializer="he_normal"))

    model.add(Conv2D(filters=256, kernel_size=(3, 3),
                     strides=1, padding="same",
                     activation="relu", kernel_initializer="he_normal"))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                           padding="same", data_format=None))

    model.add(Flatten())
    model.add(Dense(units=4096, activation='relu'))
    model.add(Dense(units=4096, activation='relu'))
    # model.add(Dense(units=num_classes, activation="softmax"))
    # 上面备注的语句也可以分开写
    model.add(Dense(num_classes))
    model.add(Activation("softmax", name="output"))

    return model

def VGGNet(input_shape, num_classes):
    # from VGG
    model = Sequential()
    # Conv1, 2
    model.add(Conv2D(kernel_size=(3, 3), activation="relu",
                     filters=64, strides=(1, 1),
                     input_shape=input_shape))

    model.add(Conv2D(kernel_size=(3, 3), activation="relu",
                     filters=64, strides=(1, 1),))

    # pool1
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

    # Conv3, 4
    model.add(Conv2D(kernel_size=(3, 3), activation="relu",
                     filters=128, strides=(1, 1)))

    model.add(Conv2D(kernel_size=(3, 3), activation="relu",
                     filters=128, strides=(1, 1)))

    # pool2
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

    # Conv5-7
    model.add(Conv2D(kernel_size=(3, 3), activation="relu",
                     filters=256, strides=(1, 1)))
    
    model.add(Conv2D(kernel_size=(3, 3), activation="relu",
                     filters=256, strides=(1, 1)))

    model.add(Conv2D(kernel_size=(3, 3), activation="relu",
                     filters=256, strides=(1, 1)))

    # pool3
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

    # Conv8-10
    model.add(Conv2D(kernel_size=(3, 3), activation="relu",
                     filters=512, strides=(1, 1)))

    model.add(Conv2D(kernel_size=(3, 3), activation="relu",
                     filters=512, strides=(1, 1)))

    model.add(Conv2D(kernel_size=(3, 3), activation="relu",
                     filters=512, strides=(1, 1)))

    # pool4
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

    # Conv11-13
    model.add(Conv2D(kernel_size=(3, 3), activation="relu",
                     filters=512, strides=(1, 1)))

    model.add(Conv2D(kernel_size=(3, 3), activation="relu",
                     filters=512, strides=(1, 1)))

    model.add(Conv2D(kernel_size=(3, 3), activation="relu",
                     filters=512, strides=(1, 1)))

    # pool5
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

    # full connected layer 1
    model.add(Flatten())
    model.add(Dense(2048))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    # full connected layer 2
    model.add(Dense(2048))
    model.add(BatchNormalization())
    model.add(Activation("relu", name="feature"))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation("softmax", name="predictions"))

    return model

# 定义基本结构, 将在Inception网络实现的具体代码中调用
def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding="same",
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN

    # Arguments:
        x: input tensor.
        filters: filters in 'Conv2D'
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in 'Conv2D'.
        strides: strides in 'Conv2D'.
        name: name of the ops; will become 'name + '_conv"
              for the convolution and 'name + '_bn" for the
              batch norm layer.

    # Returns
        Output tensor after applying 'Conv2D' and 'BatchNorm'
    """
    if name is not None:
        bn_name = name + "_bn"
        conv_name = name + "_conv"
    else:
        bn_name = None
        conv_name = None
    
    if backend.image_data_format() == "channel_first":
        bn_axis = 1
    else:
        bn_axis = 3

    x = layers.Conv2D(filters, (num_row, num_col),
                      strides=strides, padding=padding,
                      use_bias=False, name=conv_name)(x)
    x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = layers.Activation('relu', name=name)(x)

    return x

def InceptionV3(include_top=True, weights="imagenet",
                input_tensor=None, input_shape=None,
                pooling=None, classes=1000, **kwargs):
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in ("imagenet", None) or os.path.exists(weights)):
        raise valueError("The 'weights' argument should be either 'None' (random initialization), \
                         'imagenet' (pre-training on ImageNet), or the path to the weights file to be loaded.")

    if weights == "imagenet" and include_top and classes != 1000:
        raise ValueError("If using 'weights' as 'imagenet' with 'include_top' as ture, 'classes' should be 1000")

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=75,
                                      min_size=75,
                                      data_format=backend.image_data_format(),
                                      require_flatten=False,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if backend.image_data_format() == "channels_first":
        channel_axis = 1
    else:
        channel_axis = 3

    x = covn2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding="valid")
    x = conv2d_bn(x, 32, 3, 3, padding="valid")
    x = conv2d_bn(x, 64, 3, 3)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, padding="valid")
    x = conv2d_bn(x, 192, 3, 3, padding="valid")
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3db1 = conv2d_bn(x, 64, 1, 1)
    branch3x3db1 = conv2d_bn(branch3x3db1, 96, 3, 3)
    branch3x3db1 = conv2d_bn(branch3x3db1, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = layers.concatenate([branch1x1, branch5x5, branch3x3db1, branch_pool],
                           axis=channel_axis, name="mixed0")

    # mixed 1: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3db1 = conv2d_bn(x, 64, 1, 1)
    branch3x3db1 = conv2d_bn(branch3x3db1, 96, 3, 3)
    branch3x3db1 = conv2d_bn(branch3x3db1, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)

    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate([branch1x1, branch5x5, branch3x3db1, branch_pool],
                           axis=channel_axis, name="mixed1")

    # mixed 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 5, 5)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3db1 = conv2d_bn(x, 64, 1, 1)
    branch3x3db1 = conv2d_bn(branch3x3db1, 96, 3, 3)
    branch3x3db1 = conv2d_bn(branch3x3db1, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate([branch1x1, branch5x5, branch3x3db1, branch_pool],
                           axis=channel_axis, name="mixed2")

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding="valid")

    branch3x3db1 = conv2d_bn(x, 64, 1, 1)
    branch3x3db1 = conv2d_bn(branch3x3db1, 96, 3, 3)
    branch3x3db1 = conv2d_bn(branch3x3db1, 96, 3, 3, strides=(2, 2), padding="valid")

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate([branch3x3, branch3x3db1, branch_pool],
                           axis=channel_axis, name="mixed3")

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)

    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate([branch1x1, branch7x7, branch7x7db1, branch_pool],
                           axis=channel_axis, name="mixed4")

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7db1 = conv2d_bn(x, 160, 1, 1)
        branch7x7db1 = conv2d_bn(branch7x7db1, 160, 7, 1)
        branch7x7db1 = conv2d_bn(branch7x7db1, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate([branch1x1, branch7x7, branch7x7db1, branch_pool],
                               axis=channel_axis, name="mixed"+str(5+i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                           axis=channel_axis, name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding="valid")

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate([branch3x3, branch7x7x3, branch_pool],
                           axis=channel_axis, name='mixed8')

    # mixed 9, 10: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2],
                                       axis=channel_axis, name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2], 
                                          axis=channel_axis)

        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool],
                               axis=channel_axis,
                               name='mixed' + str(9 + i))

    if include_top:
        # Classification block
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        x = layers.Dense(classes, activation="softmax", name="predictions")(x)
    else:
        if pooling == "avg":
            x = layers.GloabalAveragePooling2D()(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    #Create model.
    model = models.Model(inputs, x, name="inception_v3")

    # Load weights.
    if weights == "imagenet":
        if include_top:
            weights_path = keras_utils.get_file(
                "inception_v3_weights_tf_dim_ordering_tf_kernels.h5",
                WEIGHTS_PATH,
                cache_subdir="models",
                file_hash="9a0d58056eeedaa3f26cb7ebd46da564")

        else:
            weights_path = keras_utils.get_file(
                "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",
                WEIGHT_PATH,
                cache_subdir="models",
                file_hash="bcbd6486424b2319ff4ef7d526e38f63")
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model

def tiny_XCEPTION(input_shape, name="input", l2_regularization=0.01):
    regularization = l2(l2_regularization)

    # base
    img_input = Input(input_shape, name="input")
    x = Conv2D(5, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(5, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # module 1
    residual = Conv2D(8, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(8, (3, 3), padding="same", kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(8, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = layers.add([x, residual])

    # module 2
    residual = Conv2D(16, (1, 1), strides=(2, 2), padding="same", use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # model 3
    residual = Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # model 4
    residual = Conv2D(64, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    x = Conv2D(1024, (3, 3),
               # kernel_regularizer=regularization,
               padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, name="feature")(x)
    x = Dropout(.5)(x)
    x = Dense(num_classes)(x)
    output = Activation('softmax', name='predictions')(x)
    model = Model(img_input, output)

    return model    

if __name__ == "__main__":
    input_shape = [416, 416, 3]
    num_classes = 80
    models = AlexNet(input_shape, num_classes)
    models = VGGNet(input_shape, num_classes)
    models = tiny_XCEPTION(input_shape, num_classes)
    print(models.summary())
