import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Flatten, Dense
from keras import applications

# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = 'softmax_fc_model.h5'
train_data_dir = '../Faces/MoreCategories/Train'
validation_data_dir = '../Faces/MoreCategories/Test'

n_train_man = 4500
n_train_woman = 4500
n_train_other = 4500
n_train_groups = 2250
nb_train_samples = 15750

n_validation_man = 500
n_validation_woman = 500
n_validation_other = 500
n_validation_groups = 250
nb_validation_samples = 1750

epochs = 50
batch_size = 25 


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    model.summary()

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    softmax_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save('softmax_features_train.npy', softmax_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    softmax_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save('softmax_features_validation.npy', softmax_features_validation)


def train_top_model():
    train_data = np.load('softmax_features_train.npy')
    train_labels = np.array(
        [[1, 0, 0, 0]] * n_train_man + [[0, 1, 0, 0]] * n_train_woman + [[0, 0, 1, 0]] * n_train_other + [[0, 0, 0, 1]] * n_train_groups)

    validation_data = np.load('softmax_features_validation.npy')
    validation_labels = np.array(
        [[1, 0, 0, 0]] * n_validation_man + [[0, 1, 0, 0]] * n_validation_woman + [[0, 0, 1, 0]] * n_validation_other + [[0, 0, 0, 1]] * n_validation_groups)
    
    print(len(validation_labels))
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    #TODO was 256
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    modelCheckpoint = ModelCheckpoint("faces_{epoch:02d}_{val_acc:.2f}.hdf5", monitor="val_acc", verbose=1, save_best_only=True)
    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels),
              callbacks=[modelCheckpoint])
    model.save_weights(top_model_weights_path)


save_bottlebeck_features()
train_top_model()

