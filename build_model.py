# %%
import tensorflow as tf
from tensorflow import keras
import numpy as np

# model architecture choices for decoy fmnist dataset classification
def build_model(hp):
    with tf.device('/device:GPU:0'):
        # Construct an instance of CustomModel
        inputs = keras.Input(shape=(28, 28, 1))
        x = keras.layers.Conv2D(hp.Int("filters_1", min_value=32, max_value=1024, step=32), 
                                (3, 3), 
                                activation=hp.Choice("activation_1", ["sigmoid","relu", "tanh"]),
                                padding='same', 
                                kernel_initializer= "he_uniform",
                                kernel_regularizer=tf.keras.regularizers.L1(0.01), 
                                activity_regularizer=tf.keras.regularizers.L2(0.01))(inputs)
        if hp.Boolean("BatchNormalization"):
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPool2D((2, 2), padding='same')(x)
        x = keras.layers.Flatten()(x)
        # Tune whether to use dropout.
        if hp.Boolean("dropout"):
            x = keras.layers.Dropout(0.25)(x) 

        x = keras.layers.Dense(hp.Int("fc_1", min_value=32, max_value=1024, step=32), 
                               activity_regularizer=tf.keras.regularizers.L2(0.01),
                               bias_regularizer = tf.keras.regularizers.L2(0.01))(x)
        if hp.Boolean("SecondFCLayer"):
            x = keras.layers.Dense(hp.Int("fc_2", min_value=32, max_value=1024, step=32), 
                                   activity_regularizer=tf.keras.regularizers.L2(0.01), 
                                   bias_regularizer = tf.keras.regularizers.L2(0.01))(x)
        if hp.Boolean("ThirdFCLayer"):
            x = keras.layers.Dense(hp.Int("fc_3", min_value=32, max_value=1024, step=32), 
                                   activity_regularizer=tf.keras.regularizers.L2(0.01), 
                                   bias_regularizer = tf.keras.regularizers.L2(0.01))(x)
        
        outputs = keras.layers.Dense(10, activation='softmax')(x)
        model = keras.Model(inputs, outputs)

        learning_rate = hp.Float("lr", min_value=1e-7, max_value=1e-1, sampling="log")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"]
        )
        return model

# model architecture choices for decoy coco dataset classification
def build_model_coco(hp):
    with tf.device('/device:GPU:0'):
        # Construct an instance of CustomModel
        inputs = keras.Input(shape=(224, 224, 3))
        x = keras.layers.Conv2D(hp.Int("filters_1", min_value=32, max_value=1024, step=64), 
                                (3, 3), 
                                activation=hp.Choice("activation_1", ["sigmoid","relu", "tanh"]),
                                padding='same', 
                                kernel_initializer= "he_uniform",
                                kernel_regularizer=tf.keras.regularizers.L1(0.01), 
                                activity_regularizer=tf.keras.regularizers.L2(0.01))(inputs)
        x = keras.layers.MaxPool2D((2, 2), padding='same')(x)
        x = keras.layers.Conv2D(hp.Int("filters_2", min_value=32, max_value=512, step=64), 
                                (3, 3), 
                                activation=hp.Choice("activation_2", ["sigmoid","relu", "tanh"]),
                                padding='same', 
                                kernel_initializer= "he_uniform",
                                kernel_regularizer=tf.keras.regularizers.L1(0.01), 
                                activity_regularizer=tf.keras.regularizers.L2(0.01))(x)
        x = keras.layers.MaxPool2D((2, 2), padding='same')(x)
        x = keras.layers.Conv2D(hp.Int("filters_3", min_value=32, max_value=512, step=64), 
                                (3, 3), 
                                activation="relu",
                                padding='same', 
                                kernel_initializer= "he_uniform",
                                kernel_regularizer=tf.keras.regularizers.L1(0.01), 
                                activity_regularizer=tf.keras.regularizers.L2(0.01))(x)
        x = keras.layers.MaxPool2D((2, 2), padding='same')(x)
        x = keras.layers.Conv2D(hp.Int("filters_4", min_value=32, max_value=512, step=64), 
                                (3, 3), 
                                activation="relu",
                                padding='same', 
                                kernel_initializer= "he_uniform",
                                kernel_regularizer=tf.keras.regularizers.L1(0.01), 
                                activity_regularizer=tf.keras.regularizers.L2(0.01))(x)
        x = keras.layers.MaxPool2D((2, 2), padding='same')(x)
        
        x = keras.layers.Flatten()(x)
        # Tune whether to use dropout.
        if hp.Boolean("dropout"):
            x = keras.layers.Dropout(0.35)(x)

        x = keras.layers.Dense(hp.Int("fc_1", min_value=32, max_value=1024, step=64), 
                               activity_regularizer=tf.keras.regularizers.L2(0.01),
                               bias_regularizer = tf.keras.regularizers.L2(0.01))(x)
        if hp.Boolean("SecondFCLayer"):
            x = keras.layers.Dense(hp.Int("fc_2", min_value=32, max_value=1024, step=64), 
                                   activity_regularizer=tf.keras.regularizers.L2(0.01), 
                                   bias_regularizer = tf.keras.regularizers.L2(0.01))(x)
        if hp.Boolean("ThirdFCLayer"):
            x = keras.layers.Dense(hp.Int("fc_3", min_value=32, max_value=1024, step=64), 
                                   activity_regularizer=tf.keras.regularizers.L2(0.01), 
                                   bias_regularizer = tf.keras.regularizers.L2(0.01))(x)
        
        outputs = keras.layers.Dense(2, activation='softmax')(x)
        model = keras.Model(inputs, outputs)

        learning_rate = hp.Float("lr", min_value=1e-8, max_value=1e-1, sampling="log")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"]
        )
        return model

# %% function to compute GradCAM
def compute_cam(model, i, class_idx, layer=2):    
    with tf.device('/device:GPU:0'):
        grad_model = tf.keras.models.Model([model.inputs], [model.layers[layer].output, model.output]) 

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(np.array([i]))
            loss = predictions[:, class_idx] #because image is class 1

        output = conv_outputs[0]

        grads = tape.gradient(loss, conv_outputs)[0]
        guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads
        weights = tf.reduce_mean(guided_grads, axis=(0, 1))

        cam = np.zeros(output.shape[0: 2], dtype = np.float32)
        for j, w in enumerate(weights): 
            cam += w * output[:,:,j]
        return cam
