import tensorflow as tf
import tensorflow.keras.backend as K

from config import IMAGE_SIZE
#------------------------------------COGNINET-------------------------------#

def convolutional_block(filters):
    return tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(
            filters, 3, activation='relu', padding='same'),
        tf.keras.layers.SeparableConv2D(
            filters, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D()
    ]
    )


def dense_block(units, dropout_rate):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate)
    ])

def cogni_net():
    model = tf.keras.Sequential([
            tf.keras.Input(shape=(*IMAGE_SIZE, 1)),

            tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
            tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPool2D(),

            convolutional_block(32),
            convolutional_block(64),

            convolutional_block(128),
            tf.keras.layers.Dropout(0.2),

            convolutional_block(256),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Flatten(),
            dense_block(512, 0.7),
            dense_block(128, 0.5),
            dense_block(64, 0.3),

            tf.keras.layers.Dense(
                4, activation='softmax')
        ])
    return model

#------------------------------------VGGNET-------------------------------#

def vgg_net():
    input = tf.keras.Input(shape =(*IMAGE_SIZE, 1))
    # 1st Conv Block
    x = tf.keras.layers.Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(input)
    x = tf.keras.layers.Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

    # 2nd Conv Block
    x = tf.keras.layers.Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

    # 3rd Conv block  
    x = tf.keras.layers.Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x) 
    x = tf.keras.layers.Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x) 
    x = tf.keras.layers.Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x) 
    x = tf.keras.layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

    # 4th Conv block

    x = tf.keras.layers.Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

    # 5th Conv block

    x = tf.keras.layers.Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

    # Fully connected layers  
    x = tf.keras.layers.Flatten()(x) 
    x = tf.keras.layers.Dense(units = 4096, activation ='relu')(x) 
    x = tf.keras.layers.Dense(units = 4096, activation ='relu')(x) 
    output = tf.keras.layers.Dense(units = 4, activation ='softmax')(x)

    model = tf.keras.Model (inputs=input, outputs =output)

    return model

#------------------------------------DENSENET-------------------------------#


def bn_rl_conv(x,filters,kernel=1,strides=1):
        
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, kernel, strides=strides,padding = 'same')(x)
    return x

def dense_block(x, repetition, filters=32):
        
   for _ in range(repetition):
        y = bn_rl_conv(x, 4*filters)
        y = bn_rl_conv(y, filters, 3)
        x = tf.keras.layers.concatenate([y,x])
   return x

def transition_layer(x):
        
    x = bn_rl_conv(x, K.int_shape(x)[-1] //2 )
    x = tf.keras.layers.AvgPool2D(2, strides = 2, padding = 'same')(x)
    return x

def dense_net():
    input = tf.keras.Input(shape =(*IMAGE_SIZE, 1))
    x = tf.keras.layers.Conv2D(64, 7, strides = 2, padding = 'same')(input)
    x = tf.keras.layers.MaxPool2D(3, strides = 2, padding = 'same')(x)

    for repetition in [6,12,24,16]:
        d = dense_block(x, repetition)
        x = transition_layer(d)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(d)
    output = tf.keras.layers.Dense(4, activation = 'softmax')(x)

    model = tf.keras.Model(input, output)
    return model