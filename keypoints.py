import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import PIL
import os
import PIL.ImageDraw
import PIL.Image

# set up paths to folders
def setup_paths ():
    folder_dir = "data-keypoints"
    folders = ['CAT_00', 'CAT_01', 'CAT_02', 'CAT_03', 
            'CAT_04', 'CAT_05', 'CAT_06']
    return folder_dir, folders

# displays an image with its keypoints
def display (image):
    keypoints = load_keypoints(image)
    img = mpimg.imread(image)
    imgplot = plt.imshow(img)
    plt.plot(*zip(*keypoints), marker='o', color='r', ls='')
    plt.show()
    
# create an array of images from .jpg files
def load_images ():
    folder_dir, folders = setup_paths()
    images = []
    for folder in folders:
        path = os.path.join(folder_dir, folder, '*.jpg')
        images.extend(sorted(glob.glob(path)))
    return images
    
# load keypoints from a .cat file
def load_keypoints (path):
    path += '.cat'
    
    # split the file into an array
    # array contains: number of keypoints, x1, y1, x2, y2, ...
    with open(path, 'r') as f:
        line = f.read().split()
    
    
    keypointsNumber = int(line[0])
    keypoints = []
    i = 1
    
    # fill an array with keypoints
    while i < 2 * keypointsNumber:
        keypoints.append([int(line[i]), int(line[i+1])])
        i += 2
        
    return keypoints
    

# main
images = load_images()
display(images[0])