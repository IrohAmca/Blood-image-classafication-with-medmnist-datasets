# Gerekli kütüphaneleri içe aktarın
import pandas as pd
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.utils import to_categorical
from datasets import load_dataset
from keras.optimizers import Adadelta
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# Veri yüklemesi: MedMNIST veri kümesini yükleme
train = []
test = []
vali = []
datasets = load_dataset("albertvillanova/medmnist-v2", 'bloodmnist')

train = datasets['train']
test = datasets['test']
vali = datasets['validation']

# Eğitim, test ve doğrulama verilerini alın
X_train = train['image']
y_train = train['label']

X_test = test['image']
y_test = test['label']

X_vali = vali['image']
y_vali = vali['label']

# Görüntü dönüşüm işlemleri için bir fonksiyon oluşturun
def pil_to_normalized_tensor(image_list):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Görüntüler (28, 28) boyutuna yeniden boyutlandırılır
        transforms.ToTensor(),  # Görüntüler Tensor'a dönüştürülür
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize işlemi gerçekleştirilir
    ])

    tensor_list = []
    for image in image_list:
        tensor_image = transform(image)
        tensor_list.append(tensor_image)

    return torch.stack(tensor_list)

# Görüntü verilerini tensor formatına dönüştürme ve normalize etme
X_train_array = pil_to_normalized_tensor(X_train).detach().numpy()
X_test_array = pil_to_normalized_tensor(X_test).detach().numpy()
X_vali_array = pil_to_normalized_tensor(X_vali).detach().numpy()
y_train_np = np.array(y_train)
y_test_np = np.array(y_test)
y_vali_np = np.array(y_vali)

# Görüntü verilerinin boyutunu yeniden şekillendirme
X_train_array = X_train_array.reshape(-1, 28, 28, 3)
X_vali_array = X_train_array.reshape(-1, 28, 28, 3)
X_test_array = X_test_array.reshape(-1, 28, 28, 3)

# Sınıf sayısı ve giriş boyutunu tanımlama
num_classes = 8
input_shape = (28, 28, 3)

# Model oluşturma: Evrişimli Sinir Ağı (CNN) modeli oluşturma
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
    Dense(32, activation='swish'),
    Dense(16, activation='swish'),
    Dense(num_classes, activation='softmax')
])

# Modeli derleme: Optimizasyon ve kayıp fonksiyonu tanımlama
model.summary()
model.compile(optimizer=Adadelta(learning_rate=0.35), loss='categorical_crossentropy', metrics='accuracy')

# Etiketleri kategorik hale getirme
y_train_one_hot = to_categorical(y_train_np, num_classes)
y_vali_one_hot = to_categorical(y_vali_np, num_classes)
y_test_one_hot = to_categorical(y_test_np, num_classes)

# Modeli eğitme
model.fit(X_train_array, y_train_one_hot, epochs=80)

# Modelin performansını değerlendirme: Test verileri üzerinde doğruluk ve kayıp değerlerini hesaplama
test_loss, test_accuracy = model.evaluate(X_test_array, y_test_one_hot)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)

# Eğitilen modeli kaydetme
model.save("blood_cat_90.h5")
model.save("blood_cat_90.keras")
# Tahmin işlemi için kullanıcak olan fotoğraflar bu fonksiyonn sayesinde tesnrolara dönüştürülür
def test_pil_to_tensor(index):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),       # Görüntüler (28, 28) boyutuna yeniden boyutlandırılır
        transforms.ToTensor(),             # Görüntüler Tensor'a dönüştürülür
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalizasyonu gerçekleştirilir
    ])
    tensor_image = transform(X_test[index])

    return tensor_image

# Kod ile aynı diiznde olan görselleri tensörlere çevirme
def picture_to_tensor(path):
    pil_image =Image.open(path)  
    transform = transforms.Compose([
        transforms.Resize((28, 28)),       # Görüntüler (28, 28) boyutuna yeniden boyutlandırılır
        transforms.ToTensor(),             # Görüntüler Tensor'a dönüştürülür
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalizasyonu gerçekleştirilir
    ])
    tensor_image = transform(pil_image)
    
    return tensor_image

# Görüntü tahmini işlemini fonksiyon olarak tanımlama
def tahmin(index):
    # Sınıf etiketleri ve indeksleri arasındaki eşleştirmeyi tanımlama
    türler = {
        7: "platelet",
        6: "neutrophil",
        0: "basophil",
        5: "monocyte",
        4: "lymphocyte",
        3: "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)",
        2: "erythroblast",
        1: "eosinophil"
    }
    
    # Test görüntüsünü görüntüleme
    plt.imshow(X_test[index])
    
    # Doğru etiketi alma
    label = y_test[index]
    
    # Test görüntüsünü dönüştürme ve modeli kullanarak tahmin yapma
    X_test_array_test=test_pil_to_tensor(index).detach().numpy()
    X_test_array_test=X_test_array_test.reshape(-1,28,28,3)
    tahmin_deger=model.predict(X_test_array_test)
    result_array = np.array(tahmin_deger)
    max_index=np.argmax(result_array)
    max_class=int(max_index)
    
    # Tahmin sonuçlarını ekrana yazdırma
    print("Doğru Cevap:", türler[label])
    print("Modelin Tahmini:", türler[max_class])
    
# Yerel Fotoğrafları kullanarak tahmin yapmak
def tahmin_yerel_dosya(path):
    türler = {
        7: "platelet",
        6: "neutrophil",
        0: "basophil",
        5: "monocyte",
        4: "lymphocyte",
        3: "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)",
        2: "erythroblast",
        1: "eosinophil"
    }
    image_array=picture_to_tensor(path).detach().numpy()
    image_array=image_array.reshape(-1,28,28,3)
    tahmin_deger=model.predict(image_array)
    result_array = np.array(tahmin_deger)
    max_index=np.argmax(result_array)
    max_class=int(max_index)
    print("Modelin Tahmini:", türler[max_class])
    
# Belirli bir test görüntüsü için tahmin yapmak
tahmin(290)
# Modeli eğitip kayıt ettikten sonra sadece alltaki kodu çalıştırarak modeli yüklenerek kullanılabilir
model=load_model("blood_cat_90.keras")
tahmin_yerel_dosya("images.png") 
# Not: Bu işlemi yapmak için görsel ile kodun aynı dosyada olması gerekmektedir
