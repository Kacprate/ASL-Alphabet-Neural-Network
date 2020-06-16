# ASL-Alphabet-Neural-Network
Jagiellonian University Project - image recognition neural network

## Dataset
Source: https://www.kaggle.com/grassknoted/asl-alphabet

This data set contains 87,000 images of size 200x200 pixels. There are 29 classes of which 26 are for the letters A-Z and 3 classes for SPACE, DELETE and NOTHING.

## Methods
* `__init__(self, image_witdh, image_height, output_vector_length)` - constructs the `model` class preparing it to convert all input images to 64x64 resolution. The output_vector_length is the amount of classes recognized by the neural model.
* `LoadData` is a custom method useful for data loading, preprocessing and splitting into train and test part
* `LoadModel` loads saved neural model from given path
* `CreateNeuralModel` creates a neural network model
* `Fit` fits the model with data loaded with `LoadData`
* `SaveModel` saves the neural model
* `Predict` classifies the input
* `PredictFromImage` classifies the input provided image path
* `PredictFromCV2Frame` classifies the CV2 Frame (image loaded with CV2 library)
* `Evaluate` equivalent to model.evaluate(X_test, y_test)

## Research results
The model achieved validation accuracy of 99.03% and training accuracy of 96.35% after 4 epochs. It was saved using checkpoints if an improvement in validation accuracy was made. The training process was terminated manually in 6th epoch but that epoch and the 5th one shown no improvement in comparison to the 4th epoch which is why no more checkpoints were created.

Despite the large amount of training data the model has problems with recognizing letters presented by hand on a background different than a plain white wall. What is more problems also occur when trying to differentiate letters such as R and U as they are very similiar in ASL.

Possible improvements could be made in preprocessing and more epochs could be used to train the model in order to resolve the problem mentioned lastly. However when it comes to recognition on a random background - the whole dataset should be replaced or filled with lots of new, more suitable data. Another solution is to preprocess the input of the prediction so that we extract only the the hand and change the background to white.
