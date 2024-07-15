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

# import the model
model=models.load_model('image_classifier.keras')

img=cv.imread('deer2.jpg')
img=cv.cvtColor(img,cv.COLOR_BGR2RGB)

plt.imshow(img,cmap=plt.cm.binary)

prediction=model.predict(np.array([img]) / 255)
# The prediction is an array of 10 values, which represent the model's confidence that the image corresponds to each of the 10 different classes of images. You can determine which class the model is most confident that the image belongs to by finding the index of the highest value in the prediction array
#argmax() function returns the index of the highest value in the array
index=np.argmax(prediction)
print(f'Prediction is {class_names[index]} with a confidence of {prediction[0][index]}')
plt.show()

