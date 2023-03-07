import cv2
import os
from os import listdir

# configure the mobilenet ssd model
def setup_model():
    # set up for the model: mobilenet SSD v3 using COCO database
    configPath = 'coco/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'coco/frozen_inference_graph.pb'

    # create the model
    model = cv2.dnn_DetectionModel(weightsPath,configPath)
    model.setInputSize(300,300)
    model.setInputScale(1.0/ 127.5)
    model.setInputMean((127.5, 127.5, 127.5))
    model.setInputSwapRB(True)
    
    return model

# a function to detect a cat in an image
# and draw a bounding box around it
def main ():
    model = setup_model()
    folder_dir = "data-pain"
    thres = 0.3
    
    # iterate over images
    for image in os.listdir(folder_dir):
        # read the image
        path = folder_dir + "/" + image
        img = cv2.imread(path)
        
        # path for saving images
        cat_path = "data/head/" + image
        
        # make sure the image is not Null
        if (img is None):
            continue
         
        # classify the image    
        classIds, confs, bbox = model.detect(img, confThreshold = thres)

        # if a cat is detected, create a bounding box around it
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                if (classId == 17):
                    # crop and save the image
                    cat = img[box[1]:box[3], box[0]:box[2]]
                    if (cat.shape[0] == 0) or (cat.shape[1] == 0) or(cat is None):
                        cv2.imwrite(cat_path, img)
                        continue
                    cv2.imwrite(cat_path, cat)
                    continue
        else:
            cv2.imwrite(cat_path, img)
    
        #cv2.imshow("Output",img)
        #cv2.waitKey(0)
    #cv2.waitKey(1)

main()