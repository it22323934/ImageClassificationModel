import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets,layers,models

# Load the dataset
(training_images, training_labels),(testing_images,testing_labels)=datasets.cifar10.load_data()

#individual pixel has a value from 0 to 255(pixel activation based on the brightness) so we are going to normalize the data and scale it down to 0 and 1
training_images,testing_images=training_images/255,testing_images/255

# Define class names list. The nueral network will be able to differentiate between these classes and is capable of identifying the images
class_names=['Plane','Car','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

# creating a 4x4 grid of images and assignning the class names to the images in the grid
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])    
    
plt.show()

# Selecting only a subset of the images from the training and testing dataset. Limits to  20000 images from the training dataset and 4000 images from the testing dataset.Helps to save alot of time and resources.Inorder to achieve a high accurancy you can delete the code below and train the model using the whole data set
training_images=training_images[:20000]
training_labels=training_labels[:20000]
testing_images=testing_images[:4000]
testing_labels=testing_labels[:4000]

# Create a convolutional neural network model
model=models.Sequential()
# Add the layers to the model
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
# Add the pooling layer. Filters out the essential features from the images and reduces the size of the image
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
# Flatten the model to pass it to the dense layer
model.add(layers.Flatten())
# Add the dense layers
model.add(layers.Dense(64,activation='relu'))
# Add the output layer
model.add(layers.Dense(10,activation='softmax'))

# Compile the model. Optimizer is used to reduce the loss function. Loss function is used to measure the error between the predicted output and the actual output. Metrics is used to measure the performance of the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(training_images,training_labels,epochs=10,validation_data=(testing_images,testing_labels))

loss,accuracy=model.evaluate(testing_images,testing_labels)
print(f"Loss={loss},Accuracy={accuracy}")
model.save('image_classifier.keras')


