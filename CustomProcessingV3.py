from glob import glob
from keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential,Model
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import EfficientNetB0  # Import EfficientNetB0 from the efficientnet.keras module

def class_number_and_data(train_path,test_path,augmentation):
    numberOfClass = len(glob(train_path + "/*"))

    if augmentation:
        
        datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else:
        datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_data = datagen.flow_from_directory(train_path, target_size=(224, 224), class_mode='categorical')
    test_data = datagen.flow_from_directory(test_path, target_size=(224, 224), class_mode='categorical')

    return numberOfClass, train_data, test_data

def sample_visulization(path):
    for i in os.listdir(path):
        for j in os.listdir(path + i):
            img = load_img(path + i + "/" + j)
            plt.imshow(img)
            plt.title(i)
            plt.axis("off")
            plt.show()
            break

def thelayers(model):
    for i in model:
        print(i)

def model_create(numberOfClass):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(numberOfClass, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def loss_plt(history):
    plt.plot(history.history["loss"], label = "training loss")
    plt.plot(history.history["val_loss"], label = "validation loss")
    plt.legend()
    plt.show()

def accuracy_plt(history):
    plt.plot(history.history["accuracy"], label = "accuracy")
    plt.plot(history.history["val_accuracy"], label = "validation accuracy")
    plt.legend()
    plt.show()
    

def testing(image_path,model):
    img = load_img(image_path, target_size=(224, 224,3))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 
    single_image_prediction = model.predict(img_array)
    prediction_array = single_image_prediction[0]
    return prediction_array

def mux(pred):
    index = np.array(pred).argmax()
    if index ==0:
        return "Ajloun Castle",max(pred)
    elif index == 1 or index == 2:
        return "Jerash",max(pred)
    elif index == 3:
        return "Petra",max(pred)
    elif index == 4:
        return "Roman amphitheater",max(pred)
    elif index == 5:
        return "Umm Qais",max(pred)
    elif index == 6:
        return "Wadi Rum",max(pred)
    

def plot_result(path,mux_value):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    mux_value = str(mux_value)
    height, width, _ = img.shape
    text_x = int((width - len(mux_value) * 10) / 2)  
    text_y = int(height / 2)
    font_scale = 2
    cv2.putText(img, f'{mux_value}', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 2, cv2.LINE_AA)
    plt.imshow(img)

def creat_checkpoint(name):
    checkpoint_best = ModelCheckpoint('./models/best_'+name+'.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    checkpoint_last = ModelCheckpoint('./models/last_'+name+'.h5', save_weights_only=False, verbose=1)
    return checkpoint_best,checkpoint_last    

def save_model(model, name):
    name = name
    model.save("{name}.h5")
# ... (rest of the code remains unchanged)
