import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
import pandas as pd
from keras.callbacks import CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.applications.mobilenet import MobileNet, preprocess_input

tf.random.set_seed(1234)
np.random.seed(1234)

# Переменные
BATCH_SIZE = 16
NUM_EPOCHS = 30
IMG_HEIGHT = IMG_WIDTH = 224
IMG_SIZE = 224
LOG_DIR = './log'
DEGREES = 10
SHUFFLE_BUFFER_SIZE = 1024
IMG_CHANNELS = 3
NUM_CLASSES = 3


def convert_img_to_df(dataset):
    """
    Функция для загрузки изображений

    Преобразование в DataFrame,
    чтобы после использовать функцию flow_from_dataframe

    dataset: путь к папке с данными

    return: DataFrame
    """

    img_dir = Path(dataset)
    filename = list(img_dir.glob(r'**/*.jpg'))
    label = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filename))

    filename = pd.Series(filename, name='filename').astype(str)
    label = pd.Series(label, name='class')
    img_df = pd.concat([filename, label], axis=1)
    return img_df


# Датафрейм с изображениями
data_dir = './emotions/'
img_df = convert_img_to_df(data_dir)

train, test = train_test_split(
    img_df,
    test_size=0.1,
    random_state=1234,
    stratify=img_df['class']
)

# Аугментация изображений
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    #  preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
    #  zoom_range=0.2
)

# Тестовый набор не меняется, чтобы не ухудшить точность предсказаний
val_datagen = ImageDataGenerator(rescale=1. / 255)

# изображения train будут всегда перемешиваться
train_generator = train_datagen.flow_from_dataframe(train,
                                                    target_size=(IMG_WIDTH,
                                                                 IMG_HEIGHT),
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    seed=12345,
                                                    class_mode='categorical')

test_generator = val_datagen.flow_from_dataframe(
    test,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode='categorical')


def transfer_learn(train_gen, val_gen, unfreeze_percentage, learning_rate):
    """
    Перенос обучения с модели MobileNet

    """
    mobile_net = MobileNet(input_shape=(IMG_SIZE,
                                        IMG_SIZE,
                                        IMG_CHANNELS),
                           include_top=False,
                           weights="imagenet",
                           )
    mobile_net.trainable = False
    # Тонкая настройка
    num_layers = len(mobile_net.layers)
    for layer_index in range(
            int(num_layers - unfreeze_percentage * num_layers), num_layers):
        mobile_net.layers[layer_index].trainable = True
    model_with_transfer_learning = tf.keras.Sequential([
        mobile_net,
        GlobalAveragePooling2D(),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.7),
        Dense(NUM_CLASSES, activation='softmax')
    ], )

    model_with_transfer_learning.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=["accuracy"],
    )

    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', min_delta=0.00001, patience=10)
    csv_logger = CSVLogger('colorectal-transferlearn-' + 'log.csv',
                           append=True,
                           separator=';')

    history = model_with_transfer_learning.fit(
        train_gen,
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_gen),
        validation_data=val_gen,
        callbacks=[csv_logger, earlystop_callback])

    return model_with_transfer_learning, history


unfreeze_percentage = 0.15
learning_rate = 0.001

model, history = transfer_learn(train_generator, test_generator, unfreeze_percentage, learning_rate)

model.save_weights('w_my_model.h5')
model.save('my_model.h5')
