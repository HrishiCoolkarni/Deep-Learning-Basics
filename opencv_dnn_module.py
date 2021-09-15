import cv2
import numpy as np

with open('Deep-Learning-with-OpenCV-DNN-Module\input\classification_classes_ILSVRC2012.txt','r') as f:
    img_net_class= f.read().split('\n')
    class_name= []
    for classes in img_net_class:
        class_name.append(classes.split(',')[0])
    print(class_name)
   

model= cv2.dnn.readNet(model= 'Deep-Learning-with-OpenCV-DNN-Module\input\DenseNet_121.caffemodel',
config='Deep-Learning-with-OpenCV-DNN-Module\input\DenseNet_121.prototxt',framework='Caffe')

img= cv2.imread('Deep-Learning-with-OpenCV-DNN-Module\input\image_1.jpg')
blob= cv2.dnn.blobFromImage(image= img,scalefactor=0.01, size=(224,224), mean=(104,117,123))
print(blob.shape)

model.setInput(blob)
output= model.forward()

final_prediction= output[0]

final_prediction= final_prediction.reshape(1000,1)

label_id= np.argmax(final_prediction)
class_detected= class_name[label_id]
text= "The class is: %s" %class_detected
print(class_detected)

cv2.putText(img,text,(25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow('image',img)
cv2.waitKey(0)

cv2.destroyAllWindows()

