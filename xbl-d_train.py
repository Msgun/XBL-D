# %%
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import keras_tuner
import cv2
import matplotlib.pyplot as plt
import time
import build_model 

def image_collect(path):
    tmp = []
    height, width = 224, 224
    for folder in os.listdir(path):
        class_path = path + folder + "/"
        for img in os.listdir(class_path):
            img_pth = class_path + img
            img_arr = cv2.imread(img_pth)
            img_arr = cv2.resize(img_arr, (height, width))/255
            tmp.append(img_arr)
    return tmp

x_train_con, confound_mask, x_train_clean, object_mask, x_test = [], [], [], [], []

# confounded training dataset
data_path = "./trainVszebra/training_set/confounded/"
x_train_con = image_collect(data_path)

# always clean test set
data_path = "./trainVszebra/test_set/images/"
x_test = image_collect(data_path)

# function to collect annotations in 14x14 shapes
def image_collect_14x14(path):
    # path: directory path to images
    tmp = []
    height_c, width_c = 14, 14
    for folder in os.listdir(path):
        class_path = path + folder + "/"
        for img in os.listdir(class_path):
            img_pth = class_path + img
            img_arr = cv2.imread(img_pth)
            gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
            img_arr = cv2.resize(gray, (height_c, width_c))
            img_arr = img_arr.flatten()
            img_arr = np.where(img_arr>100., 1., 0.)
            tmp.append(img_arr)
    return tmp

# confounders mask
data_path = "./trainVszebra/training_set/confounded_mask/"
confound_mask = image_collect_14x14(data_path)

# object annotations
data_path = "./trainVszebra/training_set/object_annotations/"
object_mask = image_collect_14x14(data_path)

print('images successfully loaded.')

train_size, test_size = int(len(x_train_clean)/2), int(len(x_test)/2)
y_train_arr = np.concatenate([np.zeros(train_size, dtype=np.float32),
                              np.ones(train_size, dtype=np.float32)])
y_test_arr = np.concatenate([np.zeros(test_size, dtype=np.float32),
                                  np.ones(test_size, dtype=np.float32)])
y_train_arr = to_categorical(y_train_arr)
y_test_arr = to_categorical(y_test_arr)

y_train_arr_nonencode = np.concatenate([np.zeros(train_size, dtype=np.float32),
                              np.ones(train_size, dtype=np.float32)])

# find edges of annotationX, so we only need to compare wr distance to edges instead of whole image
annotationX_tmp = object_mask.copy()
annotationX_edges = []
for i in annotationX_tmp:
    i = np.uint8(i)
    edges = cv2.Canny(i, threshold1=0.01, threshold2=0.2)
    edges = edges.flatten()
    annotationX_edges.append(edges)
    
annotationX_distance = []
for i in range(len(annotationX_edges)):
    tmp_tr = []
    for iter_, value in enumerate(annotationX_edges[i]):
        if(value>0): tmp_tr.append(iter_)
    annotationX_distance.append(tmp_tr)
    
annotationX_distance = tf.ragged.constant(annotationX_distance)
x_train_con, confound_mask, x_test, object_mask = np.array(x_train_con), np.array(confound_mask), np.array(x_test), np.array(object_mask)

# prepare train and val dataset
batch_size = 100

# training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((
    x_train_con, y_train_arr, confound_mask, object_mask, annotationX_distance))
train_dataset = train_dataset.shuffle(buffer_size=100).batch(batch_size)

# validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test_arr))
val_dataset = val_dataset.batch(batch_size)

print('dataset prepared')

# load tuned hyperparametrs from directory
tuner = keras_tuner.RandomSearch(
    hypermodel=build_model.build_model_coco,
    objective="val_accuracy",
    max_trials=60,
    executions_per_trial=2,
    overwrite=False, # so we don't overwrite tuned models
    directory="models_coco",
    project_name="tuned_class_loss",
)

# Get the top hyperparameters.
best_hps = tuner.get_best_hyperparameters()
# Build the model with the best hp.
model = build_model.build_model(best_hps[0])
# load model weights
model.load_weights("./coco.h5")

# test function
def test_step(model, x, y):
    val_logits = model(x, training=False)
    val_acc_metric[0].update_state(y, val_logits)
    val_acc_metric[1].update_state(y, val_logits)
    
# lr from keras tuner
optimizer = keras.optimizers.Adam(learning_rate= 1.7891035633842884e-05, decay=1e-6)

# Instantiate a loss function.
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
loss_cosine = tf.keras.losses.CosineSimilarity()
# Prepare the metrics.
train_acc_metric = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC(curve='PR')] 
val_acc_metric = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC(curve='PR')] 

# amount of images cam intersected with WR annotation
ctr_cam_wr_intersection = 0

def loss_fn_for_selected(model, inputs, targets, x_annotation, inputs_wr, annotation_distance):
    # Second to last var: inputs_wr (WR annotation), is to compute intersection of cam and WR annotation: 
    # resulting intersection will show if cam is on WR; then this will used to compute distance between CAM's 
    # part that is on WR and x_annotation using annotation_distance
    
    global ctr_cam_wr_intersection
    prod = 0
    for j in range(len(inputs)):

        i = inputs[j]
        ann = inputs_wr
        if(ann.dtype != tf.float32):
            ann = tf.cast(ann, tf.float32)
        class_idx = tf.math.argmax(targets[j])
        cam_orig =  build_model.compute_cam(model, i, class_idx, 8)

        cam_orig = tf.reshape(cam_orig, [-1])
        cam = tf.where(cam_orig< 0, 0., cam_orig) 
        cam = cam/(tf.reduce_max(cam) + 1e-9)

        # distance loss computation start
        wr_ann = inputs_wr[j]
        if(wr_ann.dtype != tf.float32):
            wr_ann = tf.cast(wr_ann, tf.float32)
        
        cam_wr_intersection = tf.math.multiply(cam, wr_ann)
        cam_wr_intersection = tf.where(cam_wr_intersection>0)
        
        if(len(cam_wr_intersection) > 0):
            ctr_cam_wr_intersection += 1
            tmp_wr = tf.reshape(cam_wr_intersection, [-1])
            tmp_wr = tf.cast(tmp_wr, tf.float32)
            cam_wr_ends = [tmp_wr[0], tmp_wr[-1]]
            tr_ann_distance = annotationX_distance[j]
            if(tr_ann_distance.dtype != tf.float32):
                tr_ann_distance = tf.cast(tr_ann_distance, tf.float32)
            end_one = min(tf.math.abs(tf.math.subtract(tr_ann_distance, cam_wr_ends[0])))
            end_two = min(tf.math.abs(tf.math.subtract(tr_ann_distance, cam_wr_ends[1])))
            
            # using average distance: focusing on center of gravity of object of interest in image
            average_distance = tf.math.divide(tf.math.add(end_one, end_two), 2)
            highest_distance = average_distance
            
        else:
            # there is no intersection between cam and wr
            highest_distance = 0.

        prod += highest_distance
        # distance loss end

    prod /= len(inputs)
    error = prod

    return error

# default coefficients 2.7 and 0.1 were tuned using keras tuner
def train_step_for_selected(model, x, y, x_wr, x_annotation, annotation_distance, coefs_class=2.7, coefs_exp=0.1):
    x, y, x_wr, x_annotation = tf.convert_to_tensor(x), tf.convert_to_tensor(y), tf.convert_to_tensor(x_wr), tf.convert_to_tensor(x_annotation)

    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        classification_loss = coefs_class * loss_fn(y, logits)
        explanation_loss = coefs_exp * loss_fn_for_selected(model, x, y, x_annotation, x_wr, annotation_distance)

        loss_value = (classification_loss) + (explanation_loss)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    train_acc_metric[0].update_state(y, logits)
    train_acc_metric[1].update_state(y, logits)
    return classification_loss, explanation_loss, loss_value

# %% plot accuracy history
def plot_history(history):
    plt.plot(history['train_accuracy'], label='train accuracy')
    plt.plot(history['validation_accuracy'], label='test accuracy')
    plt.plot(history['cl_loss'], label='classification loss')
    plt.plot(history['exp_loss'], label='explanation loss')
    plt.legend()
    plt.show()
    
history = {}
history['validation_accuracy'] = []
history['train_accuracy'] = []
history['train_map'] = []
history['validation_map'] = []
history['cl_loss'] = []
history['exp_loss'] = []
epochs = 100
val_acc_threshold = 0.0

for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    start_time = time.time()
    train_acc_sum, val_acc_sum = 0., 0.
    train_map_sum, val_map_sum = 0., 0.
    classification_loss, explanation_loss = 0., 0.
    
    # Iterate over the batches of the dataset.
    for step, (selected_images, selected_images_labels, lower_images_wr_mask, images_annotation, annotation_distance) in enumerate(train_dataset): #refine_dataset train_dataset
        cl_loss, exp_loss, loss_value = train_step_for_selected(model, selected_images, selected_images_labels, lower_images_wr_mask, images_annotation, annotation_distance)
        classification_loss += cl_loss
        explanation_loss += exp_loss
        train_acc = train_acc_metric[0].result()
        train_map = train_acc_metric[1].result()
    train_acc_metric[0].reset_states()
    train_acc_metric[1].reset_states()

    print("Training acc: %.4f" % (train_acc,))
    history['train_accuracy'].append((train_acc,))
    history['validation_map'].append((train_map,))
    print("Training MAP: %.4f" % (train_map,))

    print('{}/{} CAM intersected with WR.'.format(ctr_cam_wr_intersection,len(y_train_arr_nonencode)))
    # reset cam and WR intersection counter to 0 after epoch
    ctr_cam_wr_intersection = 0

    cl_loss, exp_loss = classification_loss/len(train_dataset), explanation_loss/len(train_dataset)
    print('Per epoch classification loss:%.4f' % cl_loss)
    print('per epoch explanation loss:%.6f' % exp_loss)
    history['cl_loss'].append(cl_loss)
    history['exp_loss'].append(exp_loss)
    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        test_step(model, x_batch_val, y_batch_val)
        val_acc = val_acc_metric[0].result()
        val_map = val_acc_metric[1].result()
    val_acc_metric[0].reset_states()
    val_acc_metric[1].reset_states()

    print("Validation acc: %.4f" % (float(val_acc),))
    print("Validation MAP: %.4f" % (float(val_map),))
    history['validation_accuracy'].append((val_acc,))
    history['validation_map'].append((val_map,))

    if(val_acc>val_acc_threshold):
        model.save("./models_14x14_onehot/xil_refined_models/distance_refined_weights.tf", save_format='tf') 
        model.save_weights("./models_14x14_onehot/xil_refined_models/distance_refined_weights.h5")
        val_acc_threshold = val_acc
        print('model updated')
    train_map = train_map_sum/len(train_dataset)

    print("Epoch time taken: %.2f mins" % ((time.time() - start_time)/60.0))
