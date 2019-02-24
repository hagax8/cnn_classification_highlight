import numpy as np
import keras.backend as K
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D, concatenate
from keras.models import Model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

with open("training_train.names", 'r') as file:
    xnametrain = [line.strip() for line in file]

with open("training_test.names", 'r') as file:
    xnametest = [line.strip() for line in file]

ynametrain = "train_simple.lbls"
ynametest = "test_simple.lbls"

X_train = []
X_test = []
Y_train = []
Y_test = []

with open(ynametrain, 'r') as file:
    Y_train = [line.strip() for line in file]

with open(ynametest, 'r') as file:
    Y_test = [line.strip() for line in file]

with open('train_embedding', 'r') as file:
    train_embedding = [line.strip().split(",") for line in file]

train_embedding = np.asarray(train_embedding)

with open('test_embedding', 'r') as file:
    test_embedding = [line.strip().split(",") for line in file]

test_embedding = np.asarray(test_embedding)

count = 0
for g in xnametrain:
    with open(g, 'r') as file:
        r = [line.strip().split(",") for line in file]
        if count == 0:
            X_train = r
        else:
            X_train = np.dstack((X_train,r))
        count += 1

count = 0
for g in xnametest:
    with open(g, 'r') as file:
        r = [line.strip().split(",") for line in file]
        if count == 0:
            X_test = r
        else:
            X_test = np.dstack((X_test,r))
        count += 1

X_train = np.transpose(X_train)
X_test = np.transpose(X_test)
X_train = np.expand_dims(X_train,-1)
X_test = np.expand_dims(X_test,-1)

print(X_train.shape)

#X_train=np.transpose(np.concatenate([np.transpose(X_train),np.transpose(X_train)]))
#X_test=np.transpose(np.concatenate([np.transpose(X_test),np.transpose(X_test)]))

def ProteinModel():
    auxiliary_input = Input(shape=(256,), name='aux_input')
    x = Dense(64, activation='relu')(auxiliary_input)
    main_output = Dense(1, activation='sigmoid', name='main_output')(x)
    model = Model(inputs=auxiliary_input,outputs=main_output,name='ProteinModel')
    return model

proteinModel = ProteinModel()
proteinModel.compile(optimizer="Adam", loss="binary_crossentropy",metrics=["accuracy"])
proteinModel.fit(x=train_embedding,y=Y_train,epochs=100,batch_size=30)
preds = proteinModel.evaluate(x=test_embedding,y=Y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
plot_model(proteinModel, to_file='model_withembedding_only_binary.png')

