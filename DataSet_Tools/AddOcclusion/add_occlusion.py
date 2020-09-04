import os,sys
from PIL import Image, ImageDraw
import numpy as np

images_path = "/datasets_nas/mapa3789/Pixel2Mesh/HandToolsRendered/ShapeNetHandTools_V13/"

save_path = "/datasets_nas/mapa3789/Pixel2Mesh/HandToolsRendered/ShapeNetHandTools_Occultation_Small"

occultation_factor = 5


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


def draw_ellipse_on_img(im, radius, x_start, y_start):
    draw = ImageDraw.Draw(im)
    half_rad = radius / 2
    draw.ellipse((x - half_rad, y - half_rad, x + half_rad, y + half_rad), fill=0)


def calculate_bounding_box(im):
    im.getbbox()

def calculate_mask_dimensions(bounding_box):
    pass


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
    return float(obj_pixels.size) / float(image_pixels.size)

def calc_percentage_occultation(before_ratio, after_ratio):
  return  1 - (1 / (before_ratio + 1.e-8)) * after_ratio

def get_random_point_on_object(image_pixels):
    single_vector = np.reshape(image_pixels, (1,-1))[0]
    idx_nonzero, = np.nonzero(single_vector)
    random_pixel = np.random.choice(idx_nonzero)

    y = int(np.floor(random_pixel / image_pixels.shape[0]))
    x = random_pixel % image_pixels.shape[0]

    return x,y

def export_proportions(proportions):
    export = np.asarray(proportions)

    #add mean as column
    mean_percentage_cutout = export[1:,3].astype(np.float).mean()
    export = np.insert(export, 4, mean_percentage_cutout, axis=1)
    export[0][4] = 'mean percentage cutout'

    np.savetxt(os.path.join(save_path, "proportions.csv"), export, delimiter=",", fmt="%s")
    np.savetxt("tmp/proportions.csv", export, delimiter=",", fmt="%s")
    print("FINISHED: mean percentage cutout: {}".format(mean_percentage_cutout))
    print("RUN AGAIN WITH DIFFERENT RADIUS RATIO IF NOT SATISFIED")

proportions = [['Image file', 'object to image proportion (oip)', 'oip after masking', 'percentage cutout']]
images = listFiles(images_path, ".png")


image_length = len(images)
for index, image in enumerate(images):

    print("{}/{}".format(index+1, image_length))

    im = Image.open(image).convert("RGBA")
    # bounding_box = im.getbbox()

    # replace this arbitrary number 5 with the amount you want to cut --> magic number
    # improvement: might have to adjust script, so it will always crop ~30% of the image
    # no time currently
    radius = max(im.size[0], im.size[1]) / occultation_factor

    #original image
    pixels = get_pixels(im)
    image_pixels, obj_pixels = get_image_object_pixels(pixels)
    obj_to_image_proportion_0 = get_percentage_obj_img(image_pixels, obj_pixels)

    try:
        x,y = get_random_point_on_object(image_pixels)
    except Exception:
        #we might not find a point on the object (happens if the whole image is transparent) -> should be deleted from training set
        print("ERROR: {} does not contain any information".format(image))
        continue

    draw_ellipse_on_img(im, radius, x, y)

    #img with ellipse drawn on it
    pixels = get_pixels(im)
    image_pixels, obj_pixels = get_image_object_pixels(pixels)
    obj_to_image_proportion_1 = get_percentage_obj_img(image_pixels, obj_pixels)
    percentage_occultation = calc_percentage_occultation(obj_to_image_proportion_0, obj_to_image_proportion_1)

    proportions.append([image, obj_to_image_proportion_0, obj_to_image_proportion_1, percentage_occultation])

    # should be in form of mouse/3dw_378658fc-149c-4909-b924-e003a947c69b/rendering/10.png
    suffix = os.sep.join(image.split(os.sep)[-4:])
    img_path = os.path.join(save_path, suffix)
    if not os.path.exists(os.path.dirname(img_path)):
        os.makedirs(os.path.dirname(img_path))

    im.save(img_path,"PNG")

export_proportions(proportions)
