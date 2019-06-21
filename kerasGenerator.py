import keras
import numpy as np
import cv2
import os, sys
from random import shuffle

# the DataGeneration was based on: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

class DataGenerator (keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size=32, dim=(100,200), n_channels=3, n_classes=63, shuffle=True, maxlabel=5, downsample_factor=2):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        # letters domain config 
        lower = (97, 122+1)
        upper = (65,90+1)
        nums = (48,57+1)
        iterate = [lower, upper, nums]
        self.domain = []
        self.maxlabel = maxlabel

        # domain letters for captcha
        for i in iterate:
            for j in range(i[0],i[1]):
                self.domain.append(chr(j))

    def label_to_list(self,label):
        ret = []
        for ch in label:
            ret.append(self.domain.index(ch))

        return ret

    def __len__(self):
      'Denotes the number of batches per epoch'
      return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
      'Generate one batch of data'
      # Generate indexes of the batch
      indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

      # Find list of IDs
      list_IDs_temp = [self.list_IDs[k] for k in indexes]

      # Generate data
      X, y = self.__data_generation(list_IDs_temp)

      return X, y

    def on_epoch_end(self):
      'Updates indexes after each epoch'
      self.indexes = np.arange(len(self.list_IDs))
      if self.shuffle == True:
          np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
      'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
      # Initialization
      X = np.empty((self.batch_size, *self.dim, self.n_channels))
      #y = np.empty((self.batch_size), dtype=int)
      y = np.ones((self.batch_size, 5))

      # Generate data
      for i, ID in enumerate(list_IDs_temp):
          # Store sample
          #X[i,] = np.load('gen/' + ID + '.npy')
          img = cv2.imread("generated/" + ID)#, cv2.IMREAD_GRAYSCALE)
          #img = img.astype(np.float32)
          #img = np.expand_dims(img, -1)
          X[i,] = img
          # Store class
          y[i] = [1,2,3,4,5]#self.label_to_list(self.labels[ID])
          #print("label dessa porra", y)
      inputs = {'the_input': X,
                'the_labels':y, 
                'input_lenght': np.ones((self.batch_size,1)) * self.dim[0] // self.downsample_factor - 2),
                'label_lenght': np.zeros((self.batch_size, 1))}

      return X, y#keras.utils.to_categorical(y, num_classes=self.n_classes)


'''
path = 'generated/'
partition = {'train': [], 'validation': [] } 
labels = {}
validation_size = 0.3
files_name = os.listdir(path)
files_qntd = len(files_name)
shuffle(files_name)
for filename in files_name[:int(files_qntd*validation_size)]:
    partition['validation'].append(filename)
    labels[filename] = filename[:-4]
for filename in files_name[int(files_qntd*validation_size):]:
    partition['train'].append(filename)
    labels[filename] = filename[:-4]

train_gen = DataGenerator(partition['train'], labels)
print(train_gen.__getitem__(1)[1].shape)
'''
