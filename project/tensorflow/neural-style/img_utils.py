#!/usr/bin/env python3

import os

from PIL import Image, ImageOps
import numpy as np
import scipy.misc

import matplotlib

matplotlib.rcParams['backend'] = 'TkAgg'

import matplotlib.pyplot as plt

MEAN_PIXEL = np.array([ 123.68 ,  116.779,  103.939], dtype=np.float32)

def preprocess(in_image):
    return in_image - MEAN_PIXEL

def postprocess(in_image):
    return in_image + MEAN_PIXEL

def get_image_of_size(image_path, height, width, save=True):
    image = Image.open(image_path)
    print(image.size)
    image = ImageOps.fit(image, (width, height), Image.ANTIALIAS)
    if save:
        image_name = image_path.split('/')[-1]
        out_path = '/'.join(image_path.split('/')[:-1]) + "/" + "resize_" + image_name
        if not os.path.exists(out_path):
            image.save(out_path)
    image = np.asarray(image, dtype=np.float32)
    return image

def save_image(path, image):
    # Output should add back the mean pixels we subtracted at the beginning
    image = image[0] # the image
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)

def generate_noise_image(content_image, height, width, noise_ratio=0.5):
    noise_image = np.random.uniform(-20,20, (1,height,width,3)).astype(np.float32)
    return noise_image * noise_ratio + content_image * (1 - noise_ratio)

def main():
    print(plt.style.available)
    # plt.get_backend()
    # matplotlib.rcsetup.interactive_bk
    # matplotlib.rcsetup.non_interactive_bk
    # matplotlib.rcsetup.all_backends
    plt.figure()
    image = get_image_of_size("samples/2-style2.jpg", 480, 600)
    plt.imshow(image.astype(np.uint8))
    plt.show()



if __name__ == '__main__':
    main()