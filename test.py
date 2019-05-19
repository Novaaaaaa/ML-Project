#-*-coding=utf-8-*-
import numpy as np
import cv2
from keras.models import load_model
import tensorflow as tf 
import keras.backend.tensorflow_backend as KTF  
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)
def predict(model, img):
  img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  #x = img.transpose((1, 0, 2))
  x = np.expand_dims(img, axis=0)
  print(x.shape)
  amin, amax = x.min(), x.max() 
  x = (x-amin)/(amax-amin)
  preds = model.predict(x)
  return preds[0]
# load the model
model = load_model('weight.h5')

label_map_path = "./data/label.txt"
label_map_file = open(label_map_path)
label_map = {}
for line_number, label in enumerate(label_map_file.readlines()):
    label_map[line_number] = label.split(':')[0].strip('\n')
    line_number += 1
label_map_file.close()
# local figures
img = cv2.resize(cv2.imread('0.jpg'),
                    (224, 224))
preds = predict(model,img)
ordered_index = np.argmax(preds)
print('Predicted class:',label_map[ordered_index])
print('Prediction probability:',preds[ordered_index])

