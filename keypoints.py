import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import PIL
import os
import PIL.ImageDraw

# set up paths to folders
def setup_paths ():
    folder_dir = "data-keypoints"
    folders = ['CAT_00', 'CAT_01', 'CAT_02', 'CAT_03', 
            'CAT_04', 'CAT_05', 'CAT_06']
    return folder_dir, folders

# draws keypoints on an image
def draw_keypoints(image, keypoints):
    radius = 2
    color = "blue"
    draw = PIL.ImageDraw.Draw(image)
    for x, y in keypoints:
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], color)
    return image

# displays images with their keypoints
def display_images ():
    #TO DO
    return
    
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
    #code
    return
    

# main
images = load_images()