import numpy as np
import cv2
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from os import listdir

# -----------------------

color = (67, 67, 67)  # gray

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')


def loadVggFaceModel():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))

    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    return vgg_face_descriptor


model = loadVggFaceModel()

base_output = Sequential()
base_output = Convolution2D(6, (1, 1), name='predictions')(model.layers[-4].output)
base_output = Flatten()(base_output)
base_output = Activation('softmax')(base_output)

race_model = Model(inputs=model.input, outputs=base_output)


race_model.load_weights(r'C:\sumanvitha\ethnicity_detection\race_model_single_batch.hdf5')

# ------------------------

races = ['Asian', 'Indian', 'Black', 'White', 'Middle Eastern', 'Latino_Hispanic']

# ------------------------
print('Info Starting Video Stream...............')
cap = cv2.VideoCapture(0)  # webcam

while (True):
    ret, img = cap.read()
    img = cv2.resize(img, (1200,800))
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    for (x, y, w, h) in faces:
        if w > 130:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 1)  # draw rectangle to main image

            detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face

            # add margin
            margin_rate = 30
            try:
                margin_x = int(w * margin_rate / 100)
                margin_y = int(h * margin_rate / 100)

                detected_face = img[int(y - margin_y):int(y + h + margin_y), int(x - margin_x):int(x + w + margin_x)]
                detected_face = cv2.resize(detected_face, (224, 224))  # resize to 224x224

            # display margin added face
            # cv2.rectangle(img,(x-margin_x,y-margin_y),(x+w+margin_x,y+h+margin_y),(67, 67, 67),1)

            except Exception as err:
                # print("margin cannot be added (",str(err),")")
                detected_face = img[int(y):int(y + h), int(x):int(x + w)]
                detected_face = cv2.resize(detected_face, (224, 224))

            # print("shape: ",detected_face.shape)

            if detected_face.shape[0] > 0 and detected_face.shape[1] > 0 and detected_face.shape[
                2] > 0:  # sometimes shape becomes (264, 0, 3)

                img_pixels = image.img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255

                prediction_probability = race_model.predict(img_pixels)
               # print(prediction_probability)
                prediction = np.argmax(prediction_probability)
                race = races[prediction]

               
                overlay = img.copy()
                opacity = 0.5
                cv2.rectangle(img, (x + w + 10, y - 50), (x + w + 170, y + 15), (67,67,67), cv2.FILLED)
                cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

                color = (0, 255, 0)
                probability = round(100 * prediction_probability[0][prediction], 2)

                if probability >= 51:
                    label = str(race + " (" + str(probability) + "%)")
                    cv2.putText(img, label, (int(x + w + 25), int(y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


    cv2.imshow('img', img)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# kill open cv things
cap.release()
cv2.destroyAllWindows()