import numpy as np
import cv2 
import os
from random import shuffle
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Reshape, LSTM, Input, Activation, Lambda
from keras.layers.merge import add, concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adadelta
from keras import backend as K
from kerasGenerator import DataGenerator
#from ctc_generator import TextImageGenerator

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

params = {'dim': (100,200),
          'batch_size': 64,
           #the null char
          'n_classes': 62+1,
          'n_channels': 3,
          'shuffle': True}
path = "generated"
validation_size = 0.3
'''
tiger_train = TextImageGenerator('./generated/train/', 200, 100, 64, 5)
tiger_train.build_data()

tiger_test = TextImageGenerator('./generated/train/', 200, 100, 64, 5)
tiger_test.build_data()
'''
partition = {'train': [], 'validation': [] } 
labels = {}

print("reading dataset ")
files_name = os.listdir(path)
files_qntd = len(files_name)
shuffle(files_name)
for filename in files_name[:int(files_qntd*validation_size)]:
    partition['validation'].append(filename)
    labels[filename] = filename[:-4]
for filename in files_name[int(files_qntd*validation_size):]:
    partition['train'].append(filename)
    labels[filename] = filename[:-4]
print("dataset readed!")

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

# 16 filters, (3,3) kernel_size
input_data = Input(name='my_input', shape=(100,200,1), dtype='float32')
inner = Conv2D(16, (3,3), name='conv1')(input_data)
inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)
inner = Conv2D(16, (3,3), name='conv2')(inner)
inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)
# reshape the features spliting the time and combine the features
inner = Reshape((23,768), name='reshape')(inner)
inner = Dense(10, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)

# LSTM is feeding by slices
lstm1 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(inner)
lstm1_b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(inner)
lstm1_merged = add([lstm1, lstm1_b])
lstm2 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
lstm2_b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(lstm1_merged)

# 63 is the unique tokens
inner = Dense(63, kernel_initializer='he_normal', name='dense2')(concatenate([lstm2, lstm2_b]))
y_pred = Activation('softmax', name='softmax')(inner)

'''
## para testar

model = Model(inputs=input_data, outputs=y_pred)
model.compile(optimizer='RMSprop', loss='mean_squared_error', metrics=['categorical_accuracy'])
model.summary()
print(model.input)
'''


# CTC implementation
# shape is the max len of label
labels = Input(name='the_labels',shape=[5], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

#sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
ada = Adadelta()

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada)
#model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
model.summary()
print(model.inputs)


# train test

X = np.ones((100,100,200,3), dtype='float32')
y = np.ones((100,23,63), dtype='int64')

#model.fit(X, y, epochs=100, batch_size=50, verbose=1, validation_data=(X, y))

'''
model.fit_generator(generator=tiger_train.next_batch(),
                    steps_per_epoch=int(tiger_train.n / 64),
                    epochs=30,
                    validation_data=tiger_test.next_batch(),
                    validation_steps=int(tiger_test.n / 64))
'''
# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=4)
