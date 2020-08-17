from keras.datasets import cifar10
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

from PIL import Image
import numpy as np

(train_X, train_Y), (test_X, test_Y) = cifar10.load_data() #data is already split into training and testing sets

'''showing some of the images'''
for i in range(6):
    plt.subplot(330 + 1 + i)
    plt.imshow(train_X[i])
plt.show()

'''converting the pixel values of the dataset to float types and then normalize them
Normalizing data allows the model to converge towards a solution faster (range is 0 to 1)'''
train_X = train_X.astype("float32") #have to change type to use it in next part
test_X = test_X.astype("float32")
train_X = train_X / 255.0 #normalizing the data
test_X = test_X / 255.0

'''performing one-hot encoding for target classes'''
train_Y = np_utils.to_categorical(train_Y) #converts vector to matrix
test_Y = np_utils.to_categorical(test_Y)
num_classes = test_Y.shape[1]

'''creating the sequential model and adding the layers'''
model = Sequential()
#32 filters
#(3,3) is the kernal size, always an odd number
#padding can be "valid" or "same", "same" preserves spatial dimensions so the output and input volume size match
#activation is the name of the activation you apply after performing the convolution
#kernel_constraint is the Constraint function which is applied to the kernal
#constraints is a condition of an optimization problem that the solution must satisfy
model.add(Conv2D(32, (3,3), input_shape=(32,32,3), padding="same", activation="relu",
                 kernel_constraint=maxnorm(3)))
#Dropout is a technique used to prevent overfitting
#works by setting every hidden neuron to 0 with a probability of 20%
#i.e. there is a 20% probability that the neuron will be forced to become 0
model.add(Dropout(0.2))
model.add(Conv2D((32), (3,3), activation="relu", padding="same", kernel_constraint=maxnorm(3)))
#MaxPooling reduces the spatial dimenstions of the output volume
model.add(MaxPooling2D(pool_size=(2,2)))
#flatten transforms it into a 1D array by appending each subsequent row to the one that preceded it
model.add(Flatten())
#512 neurons on this layer
model.add(Dense(512, activation="relu", kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
#softmax returns the probability that it's correct
model.add(Dense(num_classes, activation="softmax"))

'''Configure the optimizer and compile the model'''
#SGD is an optimization algorithm
sgd = SGD(lr=0.01, momentum=0.9, decay=(0.01/25), nesterov=False)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

'''viewing the model'''
model.summary()

'''Training the model'''
#might be able to improve if epochs is increased to 25
#model.fit(train_X, train_Y, validation_data=(test_X, test_Y), epochs=10, batch_size=32)
model.fit(train_X, train_Y, validation_data=(test_X, test_Y), epochs=25, batch_size=32)

'''calculating the accuracy on the test data'''
acc = model.evaluate(test_X, test_Y)
print(acc * 100)

'''saving the model'''
#model.save("model1_cifar_10epoch.h5")
model.save("model1_cifar_25epoch.h5")

'''making a dictionary to map to the output classes and make predictions from the model'''
results = {0: "Airplane", 1: "Car", 2: "Bird", 3: "Cat", 4: "Deer", 5: "Dog", 6: "Frog",
           7: "Horse", 8: "Ship", 9: "Truck"}
im = Image.open("__image_path__") #put the image path here
#the input image has to be in the shape of the dataset, i.e. (32,32,3)
im = im.resize((32,32))
im = np.expand_dims(im, axis=0)
im = np.array(im)
pred = model.predict_classes([im])[0]
print(pred, results[pred])