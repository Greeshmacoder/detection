import tensorflow as tf
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Lambda
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import Input,LeakyReLU, MaxPooling2D, AveragePooling2D, Conv2DTranspose, Conv2D, Conv1D, BatchNormalization, UpSampling2D, Add, Concatenate, SeparableConv2D, GlobalAveragePooling2D, DepthwiseConv2D, Multiply, Reshape, Maximum, Minimum, Subtract

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras import regularizers

from tensorflow.keras.optimizers import Adam, RMSprop, SGD

from tensorflow.keras import initializers

from tensorflow.keras.models import load_model,save_model

from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,LearningRateScheduler,TerminateOnNaN,ReduceLROnPlateau, TensorBoard

from tensorflow.keras import backend as K

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.resnet import ResNet50

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler,ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy

import random as rnd
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

from sklearn.metrics import classification_report, confusion_matrix

from model import make_ACFF_model


opt = tf.keras.optimizers.SGD(lr=1e-2,momentum=0.9)
loss = CategoricalCrossentropy()

#model = load_model('../results/model.h5')
inp,cls = make_ACFF_model(224,224,C=5)
model = Model(inputs=[inp], outputs=[cls])
model.load_weights('allmodel/model_weights.h5')
model.summary()
model.compile(optimizer=opt,metrics=keras.metrics.CategoricalAccuracy(),loss=loss)

def predict1(path):
    img=cv2.imread(path)
    img=cv2.resize(img,(224,224))
    img=img/255
    img=np.expand_dims(img,axis=0)
    pred=model.predict(img)[0]
    pred=np.argmax(pred)
    print(pred)
    target_names = ['collapsed_building','fire','flooded_areas','normal','traffic_incident']
    pred=target_names[pred]
    print(pred)
    return pred

if __name__=="__main__":
    from tkinter.filedialog import askopenfilename
    path=askopenfilename()
    
    predict1(path)
