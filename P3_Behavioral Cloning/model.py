import csv
import cv2
import sklearn
import numpy as np
from scipy.misc import imread
from sklearn.utils import shuffle
from keras.layers import pooling
from keras.models import Sequential 
from keras.layers import Flatten, Dense, Lambda, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers import Dropout
from keras.layers import Cropping2D
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt\

def data_generator(samples, batch_size=50, correc=0.3):
    
    '''
    samples: 
    correc: parameter to control the deviation of left and right steering angles
    batch_size: you know what is that
    '''
    
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = shuffle(samples)
    
        for start_point in range(0, num_samples, batch_size):
            batch_samples = samples[start_point : start_point + batch_size]
            
            images =[]
            angles = []
            for sample in batch_samples:
                path = './data/'
            # Create train_X
            # read in images from center, left and right cameras   
                image_center = imread(path + sample[0], mode='RGB')
                image_left = imread(path + sample[1], mode='RGB')
                image_right = imread(path + sample[2], mode='RGB')
                
                images.extend((image_center, image_left, image_right))
                
                angle_center = float(sample[3])
                angle_left = float(sample[3]) + correc
                angle_right = float(sample[3]) - correc
                
                angles.extend((angle_center, angle_left, angle_right))
            
            images = np.array(images)
            angles = np.array(angles)

            # Data augumentation: flip image in order to let it includes both clockwise and counter-clockwise images
            
            # only flip side-images:
            images_flipped = []
            angles_flipped = []
            for image, angle in zip(images, angles):
                images_flipped.append(np.fliplr(image))
                angles_flipped.append(angle*-1.0)
            images_flipped = np.array(images_flipped)
            angles_flipped = np.array(angles_flipped)
            
            train_X = np.array(np.concatenate([images, images_flipped], axis=0))
            train_Y = np.array(np.concatenate([angles, angles_flipped], axis=0))
            train_X, train_Y = sklearn.utils.shuffle(train_X, train_Y)
        
            yield train_X, train_Y
            
            
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
    lines = lines[1:]

# train, validation split
# actually we split the path but not total data, the data will be read by generator
lines = shuffle(lines)
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# compile and train the model using the generator function
correct = 0.25
batch_size = 60

train_generator = data_generator(train_samples, batch_size=batch_size, correc=correct)
validation_generator = data_generator(validation_samples, batch_size=batch_size, correc=correct)

h, w, d = 160, 320, 3
cropping_upper, cropping_bottom = 66, 20

model = Sequential()

model.add(Lambda(lambda x: (x / 255.0 -0.5), input_shape=(h, w, d)))
model.add(Cropping2D(cropping=((cropping_upper,cropping_bottom), (0,0))))
model.add(BatchNormalization())

model.add(Convolution2D(24, 5,5, subsample=(2,2), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Convolution2D(36, 5,5, subsample=(2,2), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(100))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(1))
adam = Adam(lr=0.0008)

model.compile(loss='mse', optimizer=adam)

history_object = model.fit_generator(train_generator, samples_per_epoch=
                                    len(train_samples), validation_data=
                                    validation_generator, nb_val_samples=
                                    len(validation_samples), nb_epoch=20,
                                    verbose=1)

model.save('model.h5')

### print the keys contained in the history object 
print(history_object.history.keys())

### plot the training and validation loss for epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
