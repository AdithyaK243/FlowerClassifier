import numpy as np
import tensorflow as tf
from tensorflow import keras
import random,os 
from keras.models import Sequential
from keras.layers import Conv2D, Flatten
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import cv2
from keras.utils import to_categorical
import matplotlib.pyplot as plt




#print(os.listdir(r'C:\Users\adith\Downloads\flowers (1)\flowers'))

X = []
y = []
size = 150
daisy=r'C:\Users\adith\Downloads\flowers (1)\flowers\daisy'
sunflower=r'C:\Users\adith\Downloads\flowers (1)\flowers\sunflower'
tulip=r'C:\Users\adith\Downloads\flowers (1)\flowers\tulip'
dandelion=r'C:\Users\adith\Downloads\flowers (1)\flowers\dandelion'
rose=r'C:\Users\adith\Downloads\flowers (1)\flowers\rose'

#to assing the label to images
def assign_label(img,flowerType):
    return flowerType
 
def training(flowerType,Dir):
    for img in os.listdir(Dir)[:70]:
        label=assign_label(img,flowerType)
        path = os.path.join(Dir,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        try:
            img = cv2.resize(img, (size,size))
            X.append(np.array(img))
            y.append(str(label))
        except Exception:
            pass
        

training('Daisy',daisy)
training('Sunflower',sunflower)
training('Tulip',tulip)
training('Dandelion',dandelion)
training('Rose',rose)

#to convert categorical data into binary format
labelEncoder=LabelEncoder()
Y=labelEncoder.fit_transform(y)
Y=to_categorical(Y,5)
#to make the value of input from 0->255 to 0->1
X = np.array(X)
X = X/255

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.25,random_state = 42)


np.random.seed(42)
random.seed(42)
tf.set_random_seed(42)

#Building the model
model = Sequential()
model.add(Conv2D(filters = 32,kernel_size = (5,5),activation = 'relu',input_shape = (150,150,3)))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides = (2,2)))
 

model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides = (2,2)))

model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides = (2,2)))

model.add(Flatten())#input layer
model.add(Dense(512,activation = tf.nn.relu))
model.add(Dense(5, activation = "softmax"))#output layer

model.compile(optimizer=Adam(lr = 0.001),loss='categorical_crossentropy',metrics=['accuracy'])


#fitting the data
model.fit(x_train,y_train,batch_size = 100,epochs = 10, validation_data = (x_test,y_test),verbose = 1)

loss,acc = model.evaluate(x_test,y_test)
#print(acc)

#predict
pred=model.predict(x_test)
prediction = np.argmax(pred,axis = 1)
print(pred)

print("Prediction :" + str(labelEncoder.inverse_transform(prediction)))
        


plt.imshow(x_test[0])
plt.show()