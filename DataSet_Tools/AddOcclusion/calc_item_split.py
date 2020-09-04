from PIL import Image, ImageDraw
import numpy as np
import os,sys, cv2, imutils, glob
#from skimage.metrics import structural_similarity as ssim ## for version > 0.18
from skimage.measure import compare_ssim as ssim

ground_truth_images_path = "/datasets_nas/mapa3789/Pixel2Mesh/HandToolsRendered/ShapeNetHandTools_V13/"
cropped_images_path = "/datasets_nas/mapa3789/Pixel2Mesh/HandToolsRendered/ShapeNetHandTools_Occultation_Small/"


# ground_truth_images_path = "/Users/markuspaschke/Desktop/Master_backup/Master/ShapeNetAddOccultation/TESTING/scissors_original"
# cropped_images_path = "/Users/markuspaschke/Desktop/Master_backup/Master/ShapeNetAddOccultation/TESTING/scissors_cropped"

def listFiles(dir, ext, ignoreExt=None):
    """
    Return array of all files in dir ending in ext but not ignoreExt.
    """
    matches = []
    for root, dirs, files in os.walk(dir):
        for f in files:
            if f.endswith(ext):
                if not ignoreExt or (ignoreExt and not f.endswith(ignoreExt)):
                    matches.append(os.path.join(root, f))
    return matches


# def calc_center_of_mass(file):
#     import imageio as iio
#     from skimage import filters
#     from skimage.color import rgb2gray  # only needed for incorrectly saved images
#     from skimage.measure import regionprops
#
#     image = rgb2gray(iio.imread(file))
#     threshold_value = filters.threshold_otsu(image)
#     labeled_foreground = (image > threshold_value).astype(int)
#     properties = regionprops(labeled_foreground, image)
#     center_of_mass = properties[0].centroid
#     weighted_center_of_mass = properties[0].weighted_centroid
#
#     return center_of_mass


def fill_holes(file, save_name):
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
    cv2.imwrite(save_name,im_out)

def has_item_split(ground_truth_image, cropped_image):

    # load the two input images
    imageA = cv2.imread(ground_truth_image)
    imageB = cv2.imread(cropped_image)
    # convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")

    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255,
    	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Find enclosing circles
    circles = [cv2.minEnclosingCircle(cnt) for cnt in cnts]
    # Filter for the largest one
    largest = (0, 0), 0
    for (x, y), radius in circles:
        if radius > largest[1]:
            largest = (int(x), int(y)), int(radius)

    # fill holes -> so we don't count contures which are removed after
    # the corresponding image got cropped
    fill_holes(ground_truth_image, 'filled_original.png')
    filled_original = cv2.cvtColor(cv2.imread("filled_original.png"), cv2.COLOR_BGR2GRAY)

    height,width = filled_original.shape
    mask = np.zeros((height,width), np.uint8)


    cv2.circle(mask,largest[0],largest[1] ,(255,255,255),-1)
    masked_data = cv2.bitwise_and(filled_original, filled_original, mask=mask)
    new = cv2.subtract(filled_original, masked_data)

    # amount of contours -> should represet if object is split into two parts
    ret, thresh = cv2.threshold(filled_original, 0, 255,0)
    contours,hierarchy = cv2.findContours(thresh,1,1)
    contour_length_original = len(contours)

    ret, thresh = cv2.threshold(new, 0, 255,cv2.THRESH_OTSU)
    contours,hierarchy = cv2.findContours(thresh,1,1)
    contour_length_cropped = len(contours)


    # if args.plot == True:
        # cv2.imwrite("filled_original.png", filled_original)
        # filled_original = draw_contour("filled_original.png")
        #
        # cv2.imwrite("new.png", new)
        # new = draw_contour("new.png")
        #
        # masked_data = np.repeat(masked_data[:, :, np.newaxis], 3, axis=2)
        # stack = np.concatenate((imageA, imageB, masked_data, filled_original , new), axis=1)
        # cv2.imshow('Original Image - Cropped Image - Filled Original - Mask - Filled Cropped', stack)
        # cv2.waitKey(0)

    # print(contour_length_original)
    # print(contour_length_cropped)

    return contour_length_original < contour_length_cropped


def draw_contour(file):
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
    contours,h = cv2.findContours(thresh,1,2)
    for cnt in contours:
        cv2.drawContours(img,[cnt],0,(0,0,255),1)
    return img

def save_to_pickle(splitted):
    print("Saving splitted to pickle")
    with open("tmp/item_split.dat", 'wb') as f:
        pickle.dump(splitted, f, protocol=pickle.HIGHEST_PROTOCOL)

def cleanup():
    tmp_files = glob.glob("./*.png")
    if(len(tmp_files) > 0):
        print("Removing tmp files: {}".format(" ".join(tmp_files)))
        for file in tmp_files:
            os.remove(file)

def run():
    ground_truth_images = listFiles(ground_truth_images_path, ".png")
    cropped_images = listFiles(cropped_images_path, ".png")

    if(len(ground_truth_images) == 0):
        print("No .png files found")
        sys.exit()
    elif(len(ground_truth_images) != len(cropped_images)):
        print("ground truth images and cropped images do not match (different size)")
        sys.exit()

    splitted = {}

    for index, file in enumerate(ground_truth_images):

        if (index % 50 == 0):
            print("{}/{}".format(index, len(ground_truth_images)))

        item_splitted = has_item_split(file, cropped_images[index])
        splitted[file] = item_splitted

    save_to_pickle(splitted)
    cleanup()

run()
