import pandas as pd 
import numpy as np 
from keras.datasets import mnist
import keras
from keras.models import Sequential
from keras.layers import Dense,Conv2D,pooling,Flatten,MaxPool2D,Dropout,LeakyReLU
from keras.utils import to_categorical
from datasets import load_dataset
from keras.optimizers import Adadelta,RMSprop,Adam,SGD

# Veri Hazırlama
train=[]
test=[]
vali=[]
datasets= load_dataset("albertvillanova/medmnist-v2", 'bloodmnist')

train=datasets['train']
test=datasets['test']
vali=datasets['validation']

X_train=train['image']
y_train=train['label']

X_test=test['image']
y_test=test['label']

X_vali=vali['image']
y_vali=vali['label']

import torch
import torchvision.transforms as transforms
from PIL import Image

def pil_to_normalized_tensor(image_list):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),       # Görüntüler (28, 28) boyutuna yeniden boyutlandırılır
        transforms.ToTensor(),             # Görüntüler Tensor'a dönüştürülür
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalizasyonu gerçekleştirilir
    ])

    tensor_list = []
    for image in image_list:
        tensor_image = transform(image)
        tensor_list.append(tensor_image)

    return torch.stack(tensor_list)

X_train_array=pil_to_normalized_tensor(X_train).detach().numpy()
X_test_array=pil_to_normalized_tensor(X_test).detach().numpy()
X_vali_array=pil_to_normalized_tensor(X_vali).detach().numpy()
y_train_np = np.array(y_train)
y_test_np=np.array(y_test)
y_vali_np=np.array(y_vali)
X_train_array = X_train_array.reshape(-1, 28, 28, 3)
X_vali_array = X_train_array.reshape(-1, 28, 28, 3)
X_test_array=X_test_array.reshape(-1, 28, 28, 3)

num_classes=8
input_shape = (28, 28, 3)
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), input_shape=input_shape, activation='swish'),
    MaxPool2D(pool_size=(3, 3)),
    Conv2D(64, kernel_size=(3, 3), activation='swish'),
    MaxPool2D(pool_size=(3, 3)),
    Flatten(),
    Dense(256, activation='swish'),
    Dropout(0.5),
    Dense(128, activation='swish'),
    Dense(64, activation='swish'),
    #Dropout(0.5),
    Dense(32, activation='swish'),
    #Dropout(0.3),
    Dense(16, activation='swish'),
    Dense(num_classes, activation='softmax')
])

model.summary()
model.compile(optimizer=Adadelta(learning_rate=0.35), loss='categorical_crossentropy', metrics='accuracy')

y_train_one_hot = to_categorical(y_train_np, num_classes)
y_vali_one_hot = to_categorical(y_vali_np, num_classes)
y_test_one_hot=to_categorical(y_test_np, num_classes)

# Model Eğitme
model.fit(X_train_array, y_train_one_hot,epochs=80)

# Model Test
test_loss, test_accuracy = model.evaluate(X_test_array, y_test_one_hot)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)

# from sklearn.metrics import confusion_matrix
# y_pred=model.predict(X_test_array)
# cm=confusion_matrix(y_test_one_hot, y_pred)
# print(cm)
model.save("blood_cat_88.keras")
# Görselleştirme
import matplotlib.pyplot as plt

plt.imshow(X_train[500])
plt.show()

