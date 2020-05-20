import cv2
from model import model

NNmodel = model(64, 64, 29)
NNmodel.LoadData(data_folder='./Data', load_count_images_per_class=500)
NNmodel.CreateNeuralModel()
NNmodel.Fit(validation_split=0.2, epochs=30, batch_size=32)
NNmodel.SaveModel(dir='.\\model\\')
# NNmodel.LoadModel(dir='.\\model\\')

prediction = NNmodel.Predict(NNmodel.dataset[1])
for i, p in enumerate(prediction):
    label = NNmodel.classes[p.argmax()]
    target_label = NNmodel.classes[NNmodel.dataset[3][i].argmax()]
    print("Original label: {0}; Predicted label: {1}".format(target_label, label))