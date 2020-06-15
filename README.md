# ASL-Alphabet-Neural-Network
Jagiellonian University Project - image recognition neural network

## Dataset
Source: https://www.kaggle.com/grassknoted/asl-alphabet

This data set contains 87,000 images of size 200x200 pixels. There are 29 classes of which 26 are for the letters A-Z and 3 classes for SPACE, DELETE and NOTHING.

## Methods
* `__init__(self, image_witdh, image_height, output_vector_length)` - constructs the `model` class preparing it to convert all input images to 64x64 resolution. The output_vector_length is the amount of classes recognized by the neural model.
* `LoadData` is a custom method useful for data loading
* `LoadModel` loads saved neural model from given path
* `CreateNeuralModel` creates a neural network model
* `Fit` fits the model with data loaded with `LoadData`
* `SaveModel` saves the neural model
* `Predict` classifies the input
* `PredictFromImage` classifies the input provided image path
* `PredictFromCV2Frame` classifies the CV2 Frame (image loaded with CV2 library)
* `Evaluate` equivalent to model.evaluate(X_test, y_test)
