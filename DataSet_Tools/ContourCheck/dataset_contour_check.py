"""
This Script runs over a dataset with images to find objects
containing multiple contours. For our testing purpose only single
objects and therefore single contours in the generated png's are allowed.

Use a list of images (testing set for example)

test_list_containing_dat = True
has usually the form ".../ShapeNetHandTools_V13/bolt/sn_d7bdbf351fb1d1abcd7ecf32c7a32daa/rendering/12.dat"

test_list_containing_dat = False
has usually the form ".../ShapeNetHandTools_V13/bolt/sn_d7bdbf351fb1d1abcd7ecf32c7a32daa/rendering/12.png"
"""

import os,sys, cv2, imutils, glob
import numpy as np

test_list_containing_dat = True
filelist = "/home/mapa3789/Master/Pixel2Mesh/Data/train_test_lists/HandTools_V13/test_list.txt"

def list_images():
    image_list = []
    with open(filelist, 'r') as f:
        while(True):
            line = f.readline().strip()
            if not line:
                break
            image_list.append(line)

    return image_list


def get_image_object_pixels(pixel_list):
    image_pixels = np.asarray(pixel_list)
    image_pixels = np.sum(image_pixels, axis=2) # sum color + alpha together
    obj_pixels = image_pixels[image_pixels[:,:]!=0]
    return image_pixels, obj_pixels

def get_percentage_obj_img(image_pixels, object_pixels):
    return float(object_pixels.size) / float(image_pixels.size)

def fill_holes(file):
    # Read image
    im_in = cv2.imread(file, cv2.IMREAD_GRAYSCALE);

    th, im_th = cv2.threshold(im_in, 100, 1, cv2.THRESH_BINARY_INV);

    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv
    return im_floodfill_inv


def count_contours(image):
    # fill holes -> so we don't count contures which are removed after
    # the corresponding image got cropped
    filled_image = fill_holes(image)

    # amount of contours -> should represet if object is split into two parts
    ret, thresh = cv2.threshold(filled_image, 0, 255,cv2.THRESH_OTSU)
    contours,hierarchy = cv2.findContours(thresh,1,1)
    contour_amount = len(contours) - 1

    return contour_amount



image_list = list_images()
image_length = len(image_list)
check_objects = {}


for i, image in enumerate(image_list):

    if(i % 50 == 0):
        print("{}/{}".format(i, image_length))

    # check if dat files are in test set instead of png files
    if(test_list_containing_dat):
        image = image.replace(".dat", ".png")

    # amount of contours
    contours = count_contours(image)
    if(contours > 1):
        check_objects[image[:-6].replace("/datasets_nas/mapa3789/Pixel2Mesh/HandToolsRendered/ShapeNetHandTools_V13/", "").replace("/rendering/", ""), ] = "x"



print(len(check_objects))
print(check_objects)
