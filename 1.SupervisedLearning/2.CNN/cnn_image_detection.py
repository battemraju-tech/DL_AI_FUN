#Convolutional Neural Network

#Installing Theano
#Installing TensorFlow
#Installing Keras

#Part 1 - Building the CNN

#Importing Keras librabies and packages

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import os

os.chdir('D:/Data/DataScience/DeepLearning/SuperDataScience/PART 2. CONVOLUTIONAL NEURAL NETWORKS (CNN)/Convolutional_Neural_Networks/')


#Initializing the CNN
classifier = Sequential()

#Step1: Convolution

#(32,3,3) 32 feature detectors/filters,, 3x3 F.Dectecor size. If case of GPU, can increase no.of f.dectectors.
#activation='relu', for hidden layers use activation function, relu decreases linearity between features.
#input_shape=(256,256,3) for colored-RGB images for gpu
#input_shape=(128,128,3) for colored-RGB images for cpu
#input_shape=(64,64,3) for colored-RGB images for cpu
#input_shape=(64,64,2) for black and white images for cpu.
#if use larger dimension, will increase model accuracy 
classifier.add(Convolution2D(filters=32, kernel_size=(3,3), input_shape=(64,64,3), activation='relu'))

#Step2: Max Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step3: Flattening
classifier.add(Flatten())

#Step4: Full Connection
#units=output_nodes=128
classifier.add(Dense(units=128, activation='relu'))

#sigmoid - binary outcome. here cat or dog
#softmax - more than two/binary outcome. real estate predictions. 10L, 20L, 18L, 25L....its regression outcome
classifier.add(Dense(units=1, activation='sigmoid'))

#Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

#Part 2: Fitting the CNN to images
#Images preprocessing
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                        steps_per_epoch=200, #cats=100, dogs=100images
                        epochs=10,
                        validation_data=test_set,
                        validation_steps=40)#cats=20, dogs=20

