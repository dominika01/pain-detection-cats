import cv2
import os
from os import listdir
from timeit import default_timer as timer

start = timer()

# set up for the model: mobilenet SSD v3 using COCO database
classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
 
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
folder_dir = "data"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
thres = 0.45

# result measurments
successCount = 0
count = 0
confidenceArray = []

# iterate over images
for image in os.listdir(folder_dir):
    
    # skip csv files, else read and classify the image
    if (image.endswith(".csv")):
        continue
    else:    
        path = folder_dir + "/" + image
        img = cv2.imread(path)
        classIds, confs, bbox = net.detect(img,confThreshold=thres)
        #print(classIds,bbox)

    # if a cat is detected, create a bounding box around it
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
            if (classId == 17):
                cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),1,1,(0,255,0),2)
                cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),1,1,(0,255,0),2)
                successCount += 1
                confidenceArray.append(confidence)
                break
    count += 1
    
    # display images
    #cv2.imshow("Output",img)
    #cv2.waitKey(0)

# Results
accuracy = successCount / count 
accuracy = round(accuracy, 2)
print ("Accuracy: ", accuracy)

total = 0
count = 0
for item in confidenceArray:
    total += item
    count += 1

conf = total / count
conf = round(conf, 2)
print ("Average confidence: ", conf)

end = timer()
time = (end - start) / 60
time = round(time, 2)
print ("Time in min: ", time)