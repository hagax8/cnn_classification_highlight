from keras.layers import Input, Dense, Activation, ZeroPadding2D
from keras.layers import BatchNormalization, Flatten, Conv2D, Dropout
from keras.layers import MaxPooling2D, concatenate
from keras.models import Model
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from sys import argv
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import regularizers
train_dir = argv[1]
validation_dir = argv[2]

def MultiCNN():
    main_input = Input(shape=(200, 200, 3), name='input')
    conv1 = Conv2D(32, (3, 3), activation='relu')(main_input)
    pool1 = MaxPooling2D(2,2)(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D(2,2)(conv2)
    conv3 = Conv2D(128, (3, 3), activation='relu')(pool2)
    pool3 = MaxPooling2D(2,2)(conv3)
#    conv4 = Conv2D(128, (3, 3), activation='relu')(pool3)
#    pool4 = MaxPooling2D(2,2)(conv4)
    flatt1 = Flatten()(pool3)
    drop1 = Dropout(0.5)(flatt1)
    dense1 = Dense(512, activation='relu')(drop1)
    main_output = Dense(2, activation='softmax')(dense1)
    model = Model(inputs=main_input,outputs=main_output,name='MultiCNN')
    return model

multiCNN = MultiCNN()

multiCNN.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
#    horizontal_flip=True,
#    vertical_flip=True
    )

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(200, 200),
        batch_size=20,
        class_mode= 'categorical'
        )


print(train_generator.class_indices)


validation_generator = test_datagen.flow_from_directory(
         validation_dir,
         target_size=(200, 200),
         batch_size=20,
         class_mode= 'categorical'
         )

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break


history = multiCNN.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50)

multiCNN.save('first_cnn_test_multi.h5')
train_generator.class_indices

plot_model(multiCNN, to_file='first_cnn_test_multi.png')

