import cv2
import os
import pandas as pd

# load the labels into a dataframe
def load_labels():
    print("Loading labels…")
    # read data into a pandas dataframe
    csv_path = 'data-labels/labels_preprocessed.csv'
    labels = pd.read_csv(csv_path)
    
    # replace NaN values with 0s
    labels.fillna(0, inplace=True)
    
    print("Done.\n")
    return labels
  
# load the images and labels of head  
def load_head():
    # load labels
    labels = load_labels()
    
    print("Augmenting…")
    # set up the path
    head_path = 'data-pain'
    head_dir = os.listdir(head_path)
    
    # init arrays
    i = 0
    # iterate over images
    for image in head_dir:
        # load the image
        image_path = os.path.join(head_path, image)
        img = cv2.imread(image_path)
        
        # augment
        if (img is not None):
            # check image class
            image_class = labels.loc[labels['imageid'] == image, 'head_position']
    
            if not image_class.empty:
                image_class = image_class.iloc[0]    
                
                # append an equal number of images from each class
                if (image_class == 2.0):
                    flipped = cv2.flip(img, 1)
                    path = 'data/head-flipped/flip' + str(i) + '.jpg'
                    cv2.imwrite(path, flipped)
                    i += 1
    print("Done.")
  
load_head()
  
  
  