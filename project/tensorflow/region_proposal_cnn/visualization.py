#!/usr/bin/env python

from __future__ import print_function

import math
import os

import random
import sys

import matplotlib.pyplot as plt
from construct_proposals import make_character_images

import numpy as np

import scipy.ndimage as ndimage

SIZE = 32

image = dict(make_character_images(SIZE))
#EAT the generator (key, image)

def display_character():
    char_image = np.array(image[random.choice(image.keys())])
    #print(char_image)
    plt.figure(figsize=(10,10))
    displayed_image = plt.imshow(char_image)
    displayed_image.set_cmap('hot')
    plt.show()

def main():
    display_character()
    char_image = np.array(image[random.choice(image.keys())])
    #print(char_image)
    plt.figure(figsize=(10,10))
    plt.imshow(char_image)
    plt.show()

    plt.figure(figsize=(10,10))
    plt.hist(char_image.flatten(), 256, range=(0.,1.))
    plt.show()

    img = char_image.copy()
    img = ndimage.gaussian_filter(img, sigma=(1.0,1.0), order=0)
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.show()

    plt.figure(figsize=(10,10))
    plt.hist(img.flatten(), 256, range=(0.,1.))
    plt.show()

    max_number = len(image.keys())
    number_of_columns = int(np.floor(np.sqrt(max_number))) + 1

    plt.figure(figsize=(10,10))
    for idx, x in enumerate(image.keys()):
        plt.subplot(number_of_columns, number_of_columns, idx+1)
        plt_img = np.array(image[x])
        displayed_img = plt.imshow(plt_img)
        displayed_img.set_cmap('hot')
        plt.axis('off')
    plt.show()

    exit(0)

if __name__ == '__main__':
    main()