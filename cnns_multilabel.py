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
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

class Metrics(Callback):

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(" — val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
        return

metrics = Metrics()
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
    #conv4 = Conv2D(128, (3, 3), activation='relu')(pool3)
    #pool4 = MaxPooling2D(2,2)(conv4)
    flatt1 = Flatten()(pool3)
    drop1 = Dropout(0.5)(flatt1)
    dense1 = Dense(512, activation='relu')(drop1)
    main_output = Dense(3, activation='sigmoid')(dense1)
    model = Model(inputs=main_input,outputs=main_output,name='MultiCNN')
    return model

multiCNN = MultiCNN()

multiCNN.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'],
              weighted_metrics=['acc'])

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
      steps_per_epoch=50,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50,
      class_weight={0:1.64,1:2.48,2:1}
      )
      #callbacks=[metrics])

multiCNN.save('first_cnn_test_multilabel.h5')
train_generator.class_indices

plot_model(multiCNN, to_file='first_cnn_test_multilabel.png')

