import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def get_dataset(training = True):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    if training:
        return (np.array(train_images), np.array(train_labels))
    else:
        return (test_images, test_labels)


def print_stats(images, labels):
    ankle_count = 0
    tshirt_count = 0
    trouser_count = 0
    pullover_count = 0
    dress_count = 0
    coat_count = 0
    shirt_count = 0
    sneaker_count = 0
    sandal_count = 0
    bag_count = 0
    image_count = len(images)
    image_dimension = len(images[0])
    
    for x in labels:
        if x == 0:
            tshirt_count += 1
        if x == 1:
            trouser_count += 1
        if x == 2:
            pullover_count += 1
        if x == 3:
            dress_count += 1
        if x == 4:
            coat_count += 1
        if x == 5:
            sandal_count += 1
        if x == 6:
            shirt_count += 1
        if x == 7:
            sneaker_count += 1
        if x == 8:
            bag_count += 1
        if x == 9:
            ankle_count += 1
    
    print(image_count)
    print(image_dimension, 'x', image_dimension)
    print('0. T-shirt/top -' , tshirt_count)
    print('1. Trouser -' , trouser_count)
    print('2. Pullover -' , pullover_count)
    print('3. Dress -' , dress_count)
    print('4. Coat -' , coat_count)
    print('5. Sandal -' , sandal_count)
    print('6. Shirt -' , shirt_count)
    print('7. Sneaker -' , sneaker_count)
    print('8. Bag -' , bag_count)
    print('9. Ankle boot -' , ankle_count)
    
def view_image(image, label):
    fig, ax1 = plt.subplots(1)
    ax1.set_title(label)

    ax1_projection = ax1.imshow(image)
    fig.colorbar(ax1_projection, fraction = 0.046)

    return plt.show(fig)

def build_model():
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28,28)))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(10))

    model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

    return model

def train_model(model, images, labels, T):
    model.fit(images, labels, epochs=T)
    
def evaluate_model(model, images, labels, show_loss = True):
    test_loss, test_accuracy = model.evaluate(images, labels)
    print_accuracy = round(test_accuracy, 2) * 100

    if show_loss:
        print('Loss: ' , round(test_loss,2))
        print('Accuracy: ', round(print_accuracy, 2), '%')
    else:
        print('Accuracy: ', round(print_accuracy, 2), '%')

def predict_label(model, images, index):
    model.add(keras.layers.Softmax())
    prediction = model.predict(images)
    label_list = []

    for x in range(len(prediction[index])):
        temp = (prediction[x], class_names[x])
        label_list.append(temp)
   

    print(label_list[0][0], ": ", label_list[0][1] * 100, "%")
    print(label_list[1][0], ": ", label_list[1][1] * 100, "%")
    print(label_list[2][0], ": ", label_list[2][1] * 100, "%")





