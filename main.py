import cv2
import os
from os import listdir
from timeit import default_timer as timer

# read class names for the COCO database
def get_class_names ():
    classNames= []
    classFile = 'coco.names'
    with open(classFile,'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
    return classNames

# configure the mobilenet ssd model
def setup_model_coco ():
    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'
    folder_dir = "data"
    thres = 0.45
    return configPath, weightsPath, folder_dir, thres

# calculates and displays basic metrics
def display_results (successCount, count, confidenceArray, startTime):
    # Accuracy
    accuracy = successCount / count 
    accuracy = round(accuracy, 2)
    print ("Accuracy: ", accuracy)

    # Confidence
    total = 0
    count = 0
    for item in confidenceArray:
        total += item
        count += 1

    conf = total / count
    conf = round(conf, 2)
    print ("Average confidence: ", conf)

    #Time elapsed
    endTime = timer()
    time = (endTime - startTime) / 60
    time = round(time, 2)
    print ("Time elapsed in min: ", time)
      
# a function to detect a cat in an image
# and draw a bounding box around it
def detect_cat ():
    # get the starting time
    startTime = timer()

    # set up for the model: mobilenet SSD v3 using COCO database
    classNames = get_class_names()
    configPath, weightsPath, folder_dir, thres = setup_model_coco()

    # create the model
    model = cv2.dnn_DetectionModel(weightsPath,configPath)
    model.setInputSize(320,320)
    model.setInputScale(1.0/ 127.5)
    model.setInputMean((127.5, 127.5, 127.5))
    model.setInputSwapRB(True)
    
    # result measurments
    successCount = 0
    count = 0
    confidenceArray = []

    # iterate over images
    for image in os.listdir(folder_dir):
        # read the images
        path = folder_dir + "/" + image
        img = cv2.imread(path)
        classIds, confs, bbox = model.detect(img, confThreshold = thres)
        #print(classIds,bbox)

        # if a cat is detected, create a bounding box around it
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                if (classId == 17):
                    cv2.rectangle(img, box, color = (0, 0, 255), thickness = 3)
                    cv2.putText(img, classNames[classId-1].upper(), 
                                (box[0]+10, box[1]+30), 1, 2,(0, 0, 255), 0)
                    cv2.putText(img, str(round(confidence*100,2)), 
                                (box[0]+200, box[1]+30), 1, 2, (0, 0, 255), 1)
                    successCount += 1
                    confidenceArray.append(confidence)
                    break
        count += 1
        if (count == 10):
            break
        
        # display images
        cv2.imshow("Output",img)
        cv2.waitKey(0)

    display_results(successCount, count, confidenceArray, startTime)


# main
detect_cat()