from datetime import datetime
from os import listdir

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class model:
    def __init__(self,
    image_width = 64,
    image_height = 64,
    output_vector_length = 29):
        self.image_width = image_width
        self.image_height = image_height
        self.input_shape = (image_width, image_height, 3)
        self.output_vector_length = output_vector_length

    def LoadData(self, data_folder = "./Data", load_count_images_per_class = -1):
        training_data_folder = data_folder + '/asl_alphabet_train/asl_alphabet_train'
        example_data_folder = data_folder +  '/asl_alphabet_test/asl_alphabet_test'
        
        # Data indexing
        print("Indexing data...")
        classes = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.classes = classes
        class_count = len(classes)

        # map classes to numbers
        classes_n = {}
        i = 0
        for c in classes:
            classes_n[c] = i
            i = i + 1
        data_files = {}
        data_count = 0
        for c in classes:
            data_files[c] = listdir(training_data_folder + '/' + c)
            data_count += len(data_files[c])
        print("Found {0} classes, {1} files".format(len(classes), data_count))

        # Data loading
        data = []
        labels = []
        for class_name, file_names in data_files.items():
            if load_count_images_per_class != -1:
                print("Loading " + str(load_count_images_per_class) + " images from class " + class_name)
            else:  
                print("Loading class " + class_name)
            load_count_iterator = 0
            for filename in file_names:
                if load_count_images_per_class != -1:
                    if load_count_iterator >= load_count_images_per_class:
                        break
                    load_count_iterator = load_count_iterator + 1 
                img = cv2.imread(training_data_folder + "/" + class_name + "/" + filename)
                if img is not None:
                    img = cv2.resize(img, (self.image_width, self.image_height))
                    data.append(img)
                    label_vector = np.zeros(class_count)
                    label_vector[classes_n[class_name]] = 1
                    labels.append(label_vector)

        data = np.asarray(data)
        labels = np.asarray(labels)

        X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size = 0.1)
        print("Loaded {0} data for training with set shape {1}".format(len(X_train), X_train.shape))
        print("Loaded {0} data for validation with set shape {1}".format(len(X_test), X_test.shape))
        print("Loaded {0} data labels for training with set shape {1}".format(len(Y_train), Y_train.shape))
        print("Loaded {0} data labels for validation with set shape {1}".format(len(Y_test), Y_test.shape))

        self.dataset = (X_train, X_test, Y_train, Y_test)

    def LoadModel(self, filepath):
        print('Loading model from ' + filepath)
        self.model = load_model(filepath)
        if self.model is not None:
            print('Loaded successfully')
        else:
            print('Loading error')

    def CreateNeuralModel(self):
        model = Sequential()
        model.add(Conv2D(16, kernel_size=4, strides=1, activation='relu', padding='same', input_shape=self.input_shape))
        model.add(Conv2D(32, kernel_size=4, strides=1, activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2))
        model.add(Dropout(0.1))
        model.add(Conv2D(32, kernel_size=4, strides=1, activation='relu', padding='same'))
        model.add(Conv2D(64, kernel_size=4, strides=1, activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2))
        model.add(Dropout(0.1))
        model.add(Conv2D(128, kernel_size=4, strides=1, activation='relu', padding='same'))
        model.add(Conv2D(256, kernel_size=4, strides=1, activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dropout(0.1))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.output_vector_length, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
        model.summary()
        self.model = model

    def Fit(self, epochs=30, batch_size=32):
        model_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir=".\\logs\\fit\\" + model_name
        self.model_name = model_name
        tensorboard_callback = TensorBoard(log_dir=logdir)
        if self.model is None:
            raise Exception('Model is not created!')
        if self.dataset is None:
            raise Exception('Dataset is not loaded!')

        train_image_generator = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
        )

        val_image_generator = ImageDataGenerator()

        train_generator = train_image_generator.flow(x=self.dataset[0], y=self.dataset[2], batch_size=batch_size, shuffle=True)
        val_generator = val_image_generator.flow(x=self.dataset[1], y=self.dataset[3], batch_size=batch_size, shuffle=False)
        early_stopping = EarlyStopping(patience=10,monitor="val_loss")
        checkpoint = ModelCheckpoint('.\\snapshots\\' + model_name + '\\epoch{epoch:02d}', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        self.model.fit_generator(train_generator, epochs=epochs, validation_data=val_generator, callbacks=[tensorboard_callback, early_stopping, checkpoint])

    def SaveModel(self, model_name=None):
        if model_name is None:
            model_name = self.model_name

        self.model.save('.\\models\\' + model_name)

    def Predict(self, data):
        return self.model.predict(data)

    def PredictFromImage(self, file_dir):
        img = cv2.imread(file_dir)
        if img is not None:
            img = cv2.resize(img, (self.image_width, self.image_height))
            return self.model.predict(np.asarray([img]))[0]
        else:
            print("Could not load image " + file_dir)

    def PredictFromCV2Frame(self, frame):
        if frame is not None:
            frame = cv2.resize(frame, (self.image_width, self.image_height))
            return self.model.predict(np.asarray([frame]))[0]
        else:
            print("CV2 frame not provided")

    def Evaluate(self):
        return self.model.evaluate(self.dataset[1], self.dataset[3])