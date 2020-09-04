import os,sys
from PIL import Image, ImageDraw
import numpy as np

ground_truth_images_path = "/datasets_nas/mapa3789/Pixel2Mesh/HandToolsRendered/ShapeNetHandTools_V13/"
cropped_images_path = "/datasets_nas/mapa3789/Pixel2Mesh/HandToolsRendered/ShapeNetHandTools_Occultation_Small/"


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



def get_pixels(im):
    pixels = list(im.getdata())
    width, height = im.size
    return [pixels[i * width:(i + 1) * width] for i in range(height)]

def get_image_object_pixels(pixel_list):
    image_pixels = np.asarray(pixel_list)
    image_pixels = np.sum(image_pixels, axis=2) # sum color + alpha together
    obj_pixels = image_pixels[image_pixels[:,:]!=0]
    return image_pixels, obj_pixels

def get_percentage_obj_img(image_pixels, object_pixels):
    return float(object_pixels.size) / float(image_pixels.size)

def calc_percentage_occultation(before_ratio, after_ratio):
  return  1 - (1 / (before_ratio + 1.e-8)) * after_ratio


def export_proportions(proportions):
    export = np.asarray(proportions)

    #add mean as column
    mean_percentage_cutout = export[1:,3].astype(np.float).mean()
    export = np.insert(export, 4, mean_percentage_cutout, axis=1)
    export[0][4] = 'mean percentage cutout'

    np.savetxt(os.path.join(cropped_images_path, "tmp/proportions.csv"), export, delimiter=",", fmt="%s")

    print("FINISHED: mean percentage cutout: {}".format(mean_percentage_cutout))
    print("RUN AGAIN WITH DIFFERENT RADIUS RATIO IF NOT SATISFIED")

proportions = [['Image file', 'object to image proportion (oip)', 'oip after masking', 'percentage cutout']]

def run():
    ground_truth_images = listFiles(ground_truth_images_path, ".png")
    cropped_images = listFiles(cropped_images_path, ".png")

    if(len(ground_truth_images) == 0):
        print("No .png files found")
        sys.exit()
    elif(len(ground_truth_images) != len(cropped_images)):
        print("ground truth images and cropped images do not match (different size)")
        sys.exit()


    for index, file in enumerate(ground_truth_images):

        if ((index) % 50 == 0):
            print("{}/{}".format(index, len(ground_truth_images)))

        im = Image.open(file).convert("RGBA")

        #original image
        pixels = get_pixels(im)
        image_pixels, obj_pixels = get_image_object_pixels(pixels)
        obj_to_image_proportion_0 = get_percentage_obj_img(image_pixels, obj_pixels)


        im = Image.open(cropped_images[index]).convert("RGBA")

        pixels = get_pixels(im)
        image_pixels, obj_pixels = get_image_object_pixels(pixels)
        obj_to_image_proportion_1 = get_percentage_obj_img(image_pixels, obj_pixels)
        percentage_occultation = calc_percentage_occultation(obj_to_image_proportion_0, obj_to_image_proportion_1)

        proportions.append([file, obj_to_image_proportion_0, obj_to_image_proportion_1, percentage_occultation])


run()
export_proportions(proportions)
