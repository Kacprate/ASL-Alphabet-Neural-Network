import cv2
from model import model
from termcolor import colored
import time
import numpy as np

NNmodel = model(64, 64, 29)
NNmodel.LoadData(data_folder='./data', load_count_images_per_class=5)
# NNmodel.CreateNeuralModel()
# NNmodel.Fit(epochs=20, batch_size=64)
# NNmodel.SaveModel()
NNmodel.LoadModel(filepath='.\\finalmodels\\20200610-190520')
#print(NNmodel.Evaluate())


#print("Prediction: " + NNmodel.classes[NNmodel.PredictFromImage('.\\test\\test4.jpg').argmax()])

def sortKey(x):
    return x[1]

cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    if not ret:
        print('failed to read image from camera')
        break
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # grayscale conversion

    predictionVector = np.asarray([int(x * 1000) / 1000 for x in NNmodel.PredictFromCV2Frame(frame)])
    predictionVectorWithLetters = list(zip(NNmodel.classes, predictionVector))
    print("Prediction: " + NNmodel.classes[predictionVector.argmax()])
    predictionVectorWithLetters.sort(reverse=True, key=sortKey)
    print(predictionVectorWithLetters[:3])
    print()
    time.sleep(0.3)