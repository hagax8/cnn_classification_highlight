from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten

def MolModel():
    X = Input(shape=(150,150,3), name='aux_input')
    x = Conv2D(32, (3,3), activation='relu',input_shape=(150,150,3))(X)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(64, (3,3), activation = 'relu')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(128, (3, 3), activation = 'relu')(x)
    x = MaxPooling2D(2,2)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=X,outputs=x,name='MolModel')
    return model

molModel = MolModel()
molModel.compile(optimizer="Adam", loss="binary_crossentropy",metrics=["accuracy"])
molModel.fit(x=train_embedding,y=Y_train,epochs=100,batch_size=30)
preds = proteinModel.evaluate(x=test_embedding,y=Y_test)
