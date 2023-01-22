import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import cv2
import keras_tuner
import build_model 

def image_collect(path):
    # path: train or test dirs
    tmp = []
    height, width = 224, 224
    
    for folder in os.listdir(path):
        class_path = path + folder + "/"
        for img in os.listdir(class_path):
            img_pth = class_path + img
            img_arr = cv2.imread(img_pth)
            img_arr = cv2.resize(img_arr, (height, width))
            img_arr = img_arr/255
            tmp.append(img_arr)
    return tmp

x_train, x_test = [], []

# training images with synthetically added confounding regions
data_path = "./training_set/confounded/"
x_train = image_collect(data_path)

# clean test images
data_path = "./test_set/images/"
x_test = image_collect(data_path)

print('images loaded successfully')

x_train_arr, x_test_arr = np.array(x_train), np.array(x_test)
train_size, test_size = int(len(x_train_arr)/2), int(len(x_test_arr)/2)
y_train_arr = np.concatenate([np.zeros(train_size, dtype=np.float32),
                              np.ones(train_size, dtype=np.float32)])
y_test_arr = np.concatenate([np.zeros(test_size, dtype=np.float32),
                              np.ones(test_size, dtype=np.float32)])
y_train_arr = to_categorical(y_train_arr)
y_test_arr = to_categorical(y_test_arr)

tuner = keras_tuner.RandomSearch(
    hypermodel=build_model.build_model_coco,
    objective="val_categorical_accuracy",
    max_trials=50,
    executions_per_trial=5,
    overwrite=True,
    directory="models_coco",
    project_name="tuned_class_loss",
)

# start best hyperparameter search
tuner.search(x_train_arr, y_train_arr, epochs=1, validation_data=(x_test_arr, y_test_arr))

# get best hps and best saved weights
model = tuner.get_best_models()[0]

# evaluate performnce of best model from tuning
results = model.evaluate(x_test_arr, y_test_arr, batch_size=100)
print("tuned model performance; test loss, test acc:", results)

# Train model using selected hyperparameters and save to disk
checkpoint_filepath = '.coco.h5'
epochs = 100
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

x_all = np.concatenate((x_train_arr, x_test_arr))
y_all = np.concatenate((y_train_arr, y_test_arr))

opt = keras.optimizers.Adam(learning_rate=0.0001) # replace lr with tun lr value
model.compile(loss='binary_crossentropy', optimizer=opt, 
             metrics=keras.metrics.Accuracy())

model.fit(x_all, y_all, batch_size=100, epochs=epochs, verbose=1, callbacks=[model_checkpoint_callback], validation_split=0.15)
