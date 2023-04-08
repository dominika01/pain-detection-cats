import numpy as np
import os
import pandas as pd
import cv2

print("Loading labels…")
# read data into a pandas dataframe
csv_path = 'data-labels/labels_preprocessed.csv'
labels = pd.read_csv(csv_path)

# replace NaN values with 0s
labels.fillna(0, inplace=True)

print("Done.\n")
    
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
            if (image_class == 2.0 and i<610):
                # brighten
                alpha = 1.5  # Brightness factor
                beta = 50  # Bias factor
                image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                # add contrast
                contrast = 1.5 
                mean = np.mean(image)
                image = cv2.addWeighted(image, contrast, np.zeros(image.shape, dtype=image.dtype), 0, mean * (1 - contrast))
                # add noise
                noise = np.zeros(image.shape, np.uint8)
                cv2.randn(noise, (0,0,0), (25,25,25))
                augmented = cv2.add(image, noise)

                path = 'data/head-augmented/aug' + str(i) + '.jpg'
                cv2.imwrite(path, augmented)
                i += 1
print("Done.")
print(i)