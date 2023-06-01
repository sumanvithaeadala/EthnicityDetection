
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import cv2

tqdm.pandas()

train_df = pd.read_csv("/home/201112014/fairface_label_train.csv")
test_df = pd.read_csv("/home/201112014/fairface_label_val.csv")

print("trainset consists of ",train_df.shape)
print("test set consist of ",test_df.shape)

train_df = train_df[['file', 'race']]
test_df = test_df[['file', 'race']]

train_df['file'] = '/home/201112014/fair_face/'+train_df['file']
test_df['file'] = '/home/201112014/fair_face/'+test_df['file']

train_df.head()

100*train_df.groupby(['race']).count()[['file']]/train_df.groupby(['race']).count()[['file']].sum()

# There are 2 races as East and Southeast Asian. Group them in a single Asian Race
idx = train_df[(train_df['race'] == 'East Asian') | (train_df['race'] == 'Southeast Asian')].index
train_df.loc[idx, 'race'] = 'Asian'

idx = test_df[(test_df['race'] == 'East Asian') | (test_df['race'] == 'Southeast Asian')].index
test_df.loc[idx, 'race'] = 'Asian'

100*train_df.groupby(['race']).count()[['file']]/train_df.groupby(['race']).count()[['file']].sum()

target_size = (224, 224)

def getImagePixels(file):
    #print(file)
    img = image.load_img(file, grayscale=False, target_size=target_size)
    x = image.img_to_array(img).reshape(1, -1)[0]
    return x

train_df['pixels'] = train_df['file'].progress_apply(getImagePixels)
test_df['pixels'] = test_df['file'].progress_apply(getImagePixels)

train_df.head()

train_features = []; test_features = []

for i in range(0, train_df.shape[0]):
    train_features.append(train_df['pixels'].values[i])

for i in range(0, test_df.shape[0]):
    test_features.append(test_df['pixels'].values[i])

tic = time.time()

train_features = np.array(train_features)
train_features = train_features.reshape(train_features.shape[0], 224, 224, 3)

test_features = np.array(test_features)
test_features = test_features.reshape(test_features.shape[0], 224, 224, 3)

toc = time.time()

print("converted to numpy in ",toc-tic,"seconds")

tic = time.time()

train_features = train_features / 255
test_features = test_features / 255

toc = time.time()

print("converted to numpy in ",toc-tic,"seconds")

train_label = train_df[['race']]
test_label = test_df[['race']]

races = train_df['race'].unique()

for j in range(len(races)): #label encoding
    current_race = races[j]
    print("replacing ",current_race," to ", j+1)
    train_label['race'] = train_label['race'].replace(current_race, str(j+1))
    test_label['race'] = test_label['race'].replace(current_race, str(j+1))

train_label = train_label.astype({'race': 'int32'})
test_label = test_label.astype({'race': 'int32'})

train_label.head()

train_target = pd.get_dummies(train_label['race'], prefix='race')
test_target = pd.get_dummies(test_label['race'], prefix='race')

train_target.head()

train_x, val_x, train_y, val_y = train_test_split(train_features, train_target.values
                                        , test_size=0.12, random_state=17)

import tensorflow.keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Activation
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.models import Model, Sequential

#VGG-Face model
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

model.load_weights('/home/201112014/vgg_face_weights.h5')

num_of_classes = 6  #len(races)

#freeze all layers of VGG-Face except last 7 one
for layer in model.layers[:-7]:
    layer.trainable = False

base_model_output = Sequential()
base_model_output = Convolution2D(num_of_classes, (1, 1), name='predictions')(model.layers[-4].output)
base_model_output = Flatten()(base_model_output)
base_model_output = Activation('softmax')(base_model_output)

race_model = Model(inputs=model.input, outputs=base_model_output)

race_model.compile(loss='categorical_crossentropy'
                  , optimizer=tensorflow.keras.optimizers.Adam()
                  , metrics=['accuracy']
                 )

checkpointer = ModelCheckpoint(
    filepath='race_model_single_batch.hdf5'
    , monitor = "val_loss"
    , verbose=1
    , save_best_only=True
    , mode = 'auto'
)

patience = 50

val_scores = []
train_scores = []

enableBatch = True

epochs = 30

if enableBatch != True:
    early_stop = EarlyStopping(monitor='val_loss', patience=patience) 
    
    score = race_model.fit(
        train_x, train_y
        , epochs=epochs
        , validation_data=(val_x, val_y)
        , callbacks=[checkpointer, early_stop]
    )
else:
    batch_size = pow(2, 14)
    last_improvement = 0
    best_iteration = 0
    
    loss = 1000000 #initialize as a large value
    
    for i in range(0, epochs):
        
        print("Epoch ", i, ". ", end='')
        
        ix_train = np.random.choice(train_x.shape[0], size=batch_size)
        
        score = race_model.fit(
            train_x[ix_train], train_y[ix_train]
            , epochs=1
            , validation_data=(val_x, val_y)
            , callbacks=[checkpointer]
        )
        
        val_loss = score.history['val_loss'][0]
        train_loss = score.history['loss'][0]
        
        val_scores.append(val_loss)
        train_scores.append(train_loss)
         #--------------------------------
        
        if val_loss < loss:
            loss = val_loss * 1
            last_improvement = 0
            best_iteration = i * 1
        else:
            last_improvement = last_improvement + 1
            print("try to decrease val loss for ",patience - last_improvement," epochs more")
        
        #--------------------------------
        
        if last_improvement == patience:
            print("there is no loss decrease in validation for ",patience," epochs. early stopped")
            break

if enableBatch != True:
    plt.plot(score.history['val_loss'], label='val_loss')
    plt.plot(score.history['loss'], label='train_loss')
    plt.legend(loc='upper right')
    plt.show()
else:
    plt.plot(val_scores[0:best_iteration], label='val_loss')
    plt.plot(train_scores[0:best_iteration], label='train_loss')
    plt.legend(loc='upper right')
    plt.show()

if enableBatch != True:
    plt.plot(score.history['val_loss'][0:best_iteration], label='val_loss')
    plt.plot(score.history['loss'][0:best_iteration], label='train_loss')
    plt.legend(loc='upper right')
    plt.show()
else:
    plt.plot(val_scores[0:best_iteration+1], label='val_loss')
    plt.plot(train_scores[0:best_iteration+1], label='train_loss')
    plt.legend(loc='upper right')
    plt.show()

#restore the best weights
from keras.models import load_model
race_model = load_model("race_model_single_batch.hdf5")

race_model.save_weights('/home/201112014/models/race_model_single_batch.h5')

test_perf = race_model.evaluate(test_features, test_target.values, verbose=1)
print(test_perf)

validation_perf = race_model.evaluate(val_x, val_y, verbose=1)
print(validation_perf)

#Check model is robust
f=abs(validation_perf[0] - test_perf[0]) < 0.01
if f==True:
    print('Model developed is robust')
else:
    print('Model developed isn\'t robust')

predictions = race_model.predict(test_features)

prediction_classes = []; actual_classes = []

for i in range(0, predictions.shape[0]):
    prediction = np.argmax(predictions[i])
    prediction_classes.append(races[prediction])
    actual = np.argmax(test_target.values[i])
    actual_classes.append(races[actual])
    
    if i in [
        375, 470, 750, 758, 875, 992, 1061, 2181, 2255, 4725, 4944 #latino
        , 124, 339, 762, 913, 1340, 1363, 2205 #black
        , 33, 83, 237, 609, 817, 1223, 1377 #asian
        , 109, 203, 899, 1094, 1180, 1250, 1395, 1556 #indian
        , 638, 718, 1088, 1460, 4396, 4477 #middle eastern
        , 413, 447, 573, 649, 723, 1258, 1274, 1430, 1485 #white
        , 17, 235, #misclassified
    ]:
        print(i)
        print("Actual: ",races[actual])
        print("Predicted: ",races[prediction])
        
        img = (test_df.iloc[i]['pixels'].reshape([224, 224, 3])) / 255
        plt.imshow(img)
        plt.show()
        print("----------------------")

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import seaborn as sn

cm = confusion_matrix(actual_classes, prediction_classes)
acc = accuracy_score(actual_classes, prediction_classes)
print('\n-------------------Confusion Matrix-----------\n')
print(cm)
print('\n ----------------Accuracy-------------\n')
print(acc)

df_cm = pd.DataFrame(cm, index=races, columns=races)

sn.heatmap(df_cm, annot=True,annot_kws={"size": 10})

races = ['Asian', 'Indian', 'Black', 'White', 'Middle Eastern', 'Latino_Hispanic']

demo_set = ['obama.jpg', 'sundar_pichai.jpg', 'fei_fei_li.jpg']

for file in demo_set:
    path = r'/home/201112014/tests/'+file 
    img = image.load_img(path, grayscale=False, target_size=(224, 224, 3))
    img = image.img_to_array(img).reshape(1, -1)[0]
    img = img.reshape(224, 224, 3)
    img = img / 255
    
    plt.imshow(img)
    plt.show()
    
    img = np.expand_dims(img, axis=0) 

    prediction_proba = race_model.predict(img)

    print("Prediction: ",races[np.argmax(prediction_proba)])
    print("---------------------------")






