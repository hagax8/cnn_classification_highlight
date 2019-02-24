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
from keras.applications import ResNet50

train_dir = argv[1]
validation_dir = argv[2]


def SimpleCNN():
    base_model = ResNet50(include_top=False, weights=None, input_shape=(200,200,3))
    x = base_model.output
    #pool4 = MaxPooling2D(2,2)(x)
    flatt1 = Flatten()(x)
    #drop1 = Dropout(0.05)(flatt1)
    dense1 = Dense(512, activation='relu')(flatt1)
    main_output = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=base_model.input,outputs=main_output,name='SimpleCNN')
    return base_model, model

basemodel, simpleCNN = SimpleCNN()
simpleCNN.summary()

simpleCNN.compile(loss='binary_crossentropy',
                      optimizer=optimizers.RMSprop(lr=1e-4),
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
        class_mode='binary'
        )

validation_generator = test_datagen.flow_from_directory(
         validation_dir,
         target_size=(200, 200),
         batch_size=20,
         class_mode='binary'
         )


for data_batch, labels_batch in train_generator:
     print('data batch shape:', data_batch.shape)
     print('labels batch shape:', labels_batch.shape)
     break


history = simpleCNN.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)

basemodel.save('first_cnn_test_resnet.h5')
simpleCNN.save('first_cnn_test_wholenet.h5')

plot_model(simpleCNN, to_file='first_cnn_test.png')

