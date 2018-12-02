#################################################### CNN on the MNIST data base
###### MNIST is a database of handwritten numbers
###### Libraries:

import keras
#Our handwritten character labeled dataset
from keras.datasets import mnist
from keras.models import Sequential
#Dense: fully connected layers
from keras.layers import Dense, Dropout, Flatten #Dropout: normalization process
from keras.layers import Conv2D, MaxPooling2D #Spatial convolution over images
from keras import backend as K

### Mini batch gradient descent, 128 images per batch
batch_size = 128
### 10 different characters: 0-9
num_characters = 10
### Number of epochs
epochs = 12
### Input image dimensions: 28 X 28 pixels; and are white and black thus: ONE channel
img_rows, img_cols = 28, 28

### Split the data between train and test
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#################################### 1) Analyze the data_input:

print("x_train shape: "+ str(x_train.shape), "\n")
print("x_test shape: "+ str(x_test.shape), "\n")
print("y_train shape: "+ str(y_train.shape), "\n")
print("y_test shape: "+ str(y_test.shape), "\n")
#################################### 2) Reshape our Input data

#This if statment assumes our 'data format'
#For 3D data, "channels_last" assumes: (conv_dim1, conv_dim2, conv_dim3, channels)
#while "channels_first" assumes: (channels, conv_dim1, conv_dim2, conv_dim3)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (img_rows, img_cols, 1)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

### More reshaping: change to float32 and in the range [0,1]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255 #normalize our data values in the range [0,1]
x_test /= 255 #normalize our data values in the range [0,1]

### Reshape the labels of y_train and y_test

y_train = keras.utils.to_categorical(y_train, num_characters)
y_test = keras.utils.to_categorical(y_test, num_characters)

#################################### 3) Build our model

model = Sequential()
#kernel_size= specifying the height and width of the 2D convolution window
model.add(Conv2D(filters=32, kernel_size=(3,3),
    activation='relu', input_shape= input_shape))
#Again:
model.add( Conv2D(filters= 64, kernel_size=(3, 3), activation= 'relu') )
#Chosee the best features by pooling
model.add(MaxPooling2D(pool_size=(2,2)))
#Randomly turn off neurons on and off to imporve convergence
model.add(Dropout(0.25))
#Flatten since too many dimensions, we only want a classification output
model.add(Flatten())
#Fully connected to get all relevant data
model.add(Dense(128, activation='relu'))
#Onoe more Dropout
model.add(Dropout(0.5))
# Output a 'softmax' to squash the matrix into output Probabilities
model.add(Dense(num_characters, activation= 'softmax'))

#Adaptive learning rate (adaDelta) is a popular form of gradient descent rivaled only by adam and adagrad
#categorical because: we have multiple classes (10)

model.compile(loss= keras.losses.categorical_crossentropy,
    optimizer= keras.optimizers.Adadelta(),#Adaptive gradient:Adagrad, Adadelta: prevents learning rate decay
    metrics= ['accuracy'])

### Train the model:

model.fit(x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose= 1,
    validation_data=(x_test, y_test))

#################################### 4) Evaluate the model:

score = model.evaluate(x_test, y_test, verbose=1)

print('Test loss: ', round(score[0], 4) , "\n")

print('Test accuracy: ', round(score[1], 4), "\n")

#################################### 5) Save the model:

model_json = model.to_json()

#We want to save the structure of the model itself: architecture of the model
with open("model.json", "w") as json_file:
    json_file.write(model_json)

#Save the weights (learnings)
model.save_weights("model.h5")
