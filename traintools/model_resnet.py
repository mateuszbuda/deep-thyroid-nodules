from keras.initializers import Constant
from keras import layers
from keras.layers import Input, Conv2D, Flatten, Activation, MaxPool2D, Dropout, Dense
from keras.models import Model

from focal_loss import focal_loss

img_width, img_height = 160, 160

loss_dict = {
    "out_compos": focal_loss()
}

loss_weights_dict = {
    "out_compos": 1.0,
}



def multitask_resnet():

    # modified by Adithya from 160x160x1 to 160x160x2 due to addition of nodule mask
    input_tensor = Input(shape=(img_height, img_width, 2), name="thyroid_input")
    # 160x160x8
    x = Conv2D(8, (3, 3), padding="same", activation="relu")(input_tensor)
    # 80x80x8
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    
 
    # 80x80x8
    x = residual_block(x, 8)
    x = residual_block(x, 8)
    x = residual_block(x, 12, _strides=(2,2), _project_shortcut=True)

    # 40x40x12
    x = residual_block(x, 12)
    x = residual_block(x, 12)
    x = residual_block(x, 16, _strides=(2,2), _project_shortcut=True)

    # 20x20x16
    x = residual_block(x, 16)
    x = residual_block(x, 16)
    x = residual_block(x, 24, _strides=(2,2), _project_shortcut=True)

    # 10x10x24
    x = residual_block(x, 24)
    x = residual_block(x, 24)
    x = residual_block(x, 32, _strides=(2,2), _project_shortcut=True)

    # 5x5x32
    x = residual_block(x, 32)
    x = residual_block(x, 32)
    x = residual_block(x, 32, _strides=(2,2), _project_shortcut=True)

    # 5x5x32
    x = Dropout(0.5)(x)


    y_compos = Conv2D(filters=3, kernel_size=(3, 3), kernel_initializer="glorot_normal", bias_initializer=Constant(value=-0.9))(x)
    y_compos = Flatten()(y_compos)
    y_compos = Activation("softmax", name="out_compos")(y_compos)

    return Model(inputs=[input_tensor], outputs=[y_compos])




def residual_block(y, nb_channels, _strides=(1, 1), _project_shortcut=False):
    shortcut = y

    # down-sampling is performed with a stride of 2
    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)

    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
    y = layers.BatchNormalization()(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # projection shortcut is used to match dimensions with 1x1 convolutions
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = layers.Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    y = layers.add([shortcut, y])
    y = layers.LeakyReLU()(y)

    return y
































