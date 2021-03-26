import cv2
import numpy as np

cap = cv2.VideoCapture(0)
whT = 320 #target weight and height of image
confThreshold = 0.5
nmsThreshold = 0.3

classesFile = "coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfig = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

#modelConfig = 'yolov3-tiny.cfg'
#modelWeights = 'yolov3-tiny.weights'

net = cv2.dnn.readNetFromDarknet(modelConfig,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:] # remove first 5 elements of output list
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2]*wT), int(det[3]*hT)
                x, y = int(det[0]*wT - w/2), int(det[1]*hT - h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)



while True:
    success, img = cap.read()

    blob = cv2.dnn.blobFromImage(img, 1/255, (whT,whT), [0,0,0],1,crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()] # getting names of output layers

    outputs = net.forward(outputNames)
    #print(outputs[0].shape) # (300, 85) number of bounding boxes, classes+box weight-height and confidence
    #print(outputs[1].shape) # (1200, 85)
    #print(outputs[2].shape) # (4800, 85)

    findObjects(outputs,img)

    cv2.imshow('Output',img)
    cv2.waitKey(1)
