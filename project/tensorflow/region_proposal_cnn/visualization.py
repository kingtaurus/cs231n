#!/usr/bin/env python

from __future__ import print_function

import math
import os

import random
import sys

import matplotlib.pyplot as plt
from construct_proposals import make_character_images
from construct_proposals import generate_plate
from construct_proposals import generate_plate_alt
from construct_proposals import generate_bg
from construct_proposals import generate_proposal

import numpy as np
import cv2

import scipy.ndimage as ndimage

from skimage import data
from skimage.transform import rotate, warp
from skimage.transform import AffineTransform, SimilarityTransform

SIZE = 32

image = dict(make_character_images(SIZE))
#EAT the generator (key, image)

cam_image = data.camera()

def display_character(in_char = None, figsize=(10,10)):
    """ Displays a random character
     (1) the character images are stored in image (dictionary)
         image['c'] -> numpy.ndarray
    """
    character = in_char
    if in_char is None:
        character = random.choice(image.keys())

    char_image = np.array(image[character])
    plt.figure(figsize=figsize)
    displayed_image = plt.imshow(char_image)
    displayed_image.set_cmap('hot')
    plt.axis('off')
    plt.show()
    return character

def display_histogram(in_char = None, figsize=(10,10)):
    character = in_char
    if in_char is None:
        character = random.choice(image.keys())
    char_image = np.array(image[character])
    plt.figure(figsize=figsize)
    plt.hist(char_image.flatten(), 256, range=(0.,1.))
    plt.title("Character [%s] Histogram" % character)
    plt.show()
    return character

def display_all_characters(figsize=(10,10)):
    return 0

def display_all_histograms(figsize=(10,10)):
    return 0

def main():
    display_character()
    display_histogram()
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

    plt.figure(figsize=(10,10))
    code, img, plate_mask = generate_plate(image)
    plt.imshow(img)
    plt.show()

    #using the new code ('w') corresponds to a wide space
    plt.figure(figsize=(10,10))
    plt.title("UW55wKXJ")
    code, img, plate_mask = generate_plate(image,"UW55wKXJ")
    plt.imshow(img)
    plt.show()

    #using the new code ('d') corresponds to a double space
    plt.figure(figsize=(10,10))
    plt.title("GR40dSEU")
    plt.imshow(generate_plate(image,"GR40dSEU")[1])
    plt.show()

    #using the new code ('q') corresponds to a quad space
    plt.figure(figsize=(10,10))
    plt.title("VB57qWXV")
    plt.imshow(generate_plate(image,"VB57qWXV")[1])
    plt.show()

    characters, img, plate_mask = generate_plate_alt()
    plt.figure(figsize=(10,10))
    plt.title(characters)
    plt.imshow(img)
    plt.show()

    img = cv2.imread('/usr/share/backgrounds/Beach_by_Renato_Giordanelli.jpg',0)
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

    #scikit image manipulation
    print(rotate(cam_image, 2).shape)
    plt.figure(figsize=(10,10))
    plt.subplot(3,1,1)
    plt.imshow(cam_image, cmap='gray')
    plt.subplot(3,1,2)
    plt.imshow(rotate(cam_image, 2, resize=True), cmap='gray')
    print(rotate(cam_image, 2, resize=True).shape)
    plt.subplot(3,1,3)
    plt.imshow(rotate(cam_image, 90, resize=True), cmap='gray')
    print(rotate(cam_image, 90, resize=True).shape)
    plt.show()

    #skimage.transform.swirl
    #skimage.transform.warp
    #from skimage.transform import SimilarityTransform
    #tform = SimilarityTransform(translation=(0, -10))
    #warped = warp(image, tform)

    #def shift_down(xy):
    #...xy[:, 1] -= 10
    #...return xy
    #warped = warp(image, shift_down)
    #scikit-image skeletonize
    #scikit-image.transform.AffineTransform
    plt.figure(figsize = (10,10))
    img = generate_bg()
    plt.imshow(img)
    plt.show()

    # plt.figure(figsize=(10,10))
    # affine = AffineTransform(rotation=0.1, shear=0.5, scale=(10.,10.))
    # plt.imshow(warp(cam_image, affine), cmap='gray')
    # plt.show()
    # affine transformation (note, parameterization is different than a full
    # 3D rotation matrix)
    return 0

if __name__ == '__main__':
    plt.figure(figsize = (10,10))
    img, code = generate_proposal(image)
    plt.imshow(img)
    plt.show()
    exit(0)
    main()
