#!/usr/bin/env python

#based upon the code:
#https://github.com/matthewearl/deep-anpr/blob/master/gen.py

""" 
    Generates an image which contrain an affine transformed 
    Licence Plate.
"""

from __future__ import print_function

import math
import os
import random
import sys

import cv2
import numpy as np

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from skimage.transform import AffineTransform
from skimage.transform import resize

import utils.common as common

import glob

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

OUTPUT_SHAPE = (64,128)
FONT_HEIGHT = 32
FONT_PATH   = "UKNumberPlate.ttf"

CHARS = common.CHARS + " "

ld_code = { 'L' : common.LETTERS,
            'D' : common.DIGITS,
            'S' : ' ',
            'X' : 'd',
            'Q' : 'q',
            'W' : 'w',
            'M' : 'm' }

''' ld_code: L <--> LETTERS;
             D <--> DIGITS;
             S <--> Space;
             X <--> Double Space;
             Q <--> Quad Space;
             W <--> Wide Space;
             M <--> Micro Space; (?)
'''

def make_character_images(output_height):
    font_size = output_height * 4
    font   = ImageFont.truetype(FONT_PATH, font_size)
    height = max(font.getsize(c)[1] for c in CHARS)
    #require the height to always be the same

    for c in CHARS:
        width = font.getsize(c)[0]
        im = Image.new("RGBA", (width, height), (0, 0, 0))

        draw = ImageDraw.Draw(im)
        draw.text((0, 0), c, (255, 255, 255), font=font)
        scale = float(output_height) / height
        im = im.resize((int(width * scale), output_height), Image.ANTIALIAS)
        yield c, np.array(im)[:, :, 0].astype(np.float32) / 255.

    #double space
    wide_space = "d"
    width = font.getsize(' ')[0]
    im = Image.new("RGBA", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    draw.text((0, 0), c, (255, 255, 255), font=font)
    scale = float(output_height) / height
    im = im.resize((int(width * 2.0 * scale), output_height), Image.ANTIALIAS)
    yield wide_space, np.array(im)[:, :, 0].astype(np.float32) / 255.

    #wide space
    wide_space = "w"
    width = font.getsize(' ')[0]
    im = Image.new("RGBA", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    draw.text((0, 0), c, (255, 255, 255), font=font)
    scale = float(output_height) / height
    im = im.resize((int(width * 1.5 * scale), output_height), Image.ANTIALIAS)
    yield wide_space, np.array(im)[:, :, 0].astype(np.float32) / 255.

    #quad space
    wide_space = "q"
    width = font.getsize(' ')[0]
    im = Image.new("RGBA", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    draw.text((0, 0), c, (255, 255, 255), font=font)
    scale = float(output_height) / height
    im = im.resize((int(width * 4.0* scale), output_height), Image.ANTIALIAS)
    yield wide_space, np.array(im)[:, :, 0].astype(np.float32) / 255.

    #micro space (0.5)
    wide_space = "m"
    width = font.getsize(' ')[0]
    im = Image.new("RGBA", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    draw.text((0, 0), c, (255, 255, 255), font=font)
    scale = float(output_height) / height
    im = im.resize((int(width * 0.5 * scale), output_height), Image.ANTIALIAS)
    yield wide_space, np.array(im)[:, :, 0].astype(np.float32) / 255.

    #CAN add extra characters after the fact here
    #(1) for example wide space would take ' ' and widen it to have some
    #    some larger width np.zeros(size=(y,x))
    #(2) characters like '-' might also be available

def generate_code(lp_code = None):
    """
    generate_code(lp_code = None) generate a simple "RANDOM" license plate code

      takes either a string descriptor
      with the following format:
        "L", "D", "S": L corresponds to letters;
                       D corresponds to digits;
                       S corresponds to spaces;
      there is no length requirement;

      if None (or nothing is passed in):
        it defaults to "LLDDSLL"
      returns a string of characters of the correct format;
    """
    #licence descriptor
    ld = lp_code
    if lp_code is None:
        ld = "LLDDSLLL"

    #join together a comprehension
    plate = ''.join([ random.choice(ld_code[c]) for c in ld ])
    return plate


#the code below can be updated in the following manner:
# (1) add the ability to define the padding (pad_left, pad_right, pad_top, pad_bottom)
# (2) add the ability to left, right align relative to a fixed size 'plate'
#     or maximum sized plate:
#     AB33 ABAC
#     GREG
#          KING
# CENTRAL alignment should be neglected (for now)
# (3) alpha channel? (letters would have alpha = 1)
#     background could then be textured/shadowed gradient in addition to colored;
# (4) additional fonts (i.e. each plate might have a font);
# (5) additional information on the plate.
# (6) addition of filtering the plate (i.e. smoothing out the edges, blurring etc.)

def generate_plate_alt(output_height=16, pad=(5,5,5,5), characters=None):
    top_pad, bottom_pad, left_pad, right_pad = pad

    if characters is None:
        characters = generate_code()

    font_size = output_height * 4

    radius = 1 + int(FONT_HEIGHT * 0.1 * random.random())

    font   = ImageFont.truetype(FONT_PATH, font_size)
    height = max(font.getsize(c)[1] for c in characters)
    width  = np.sum([font.getsize(c)[0] for c in characters])

    #require the height to always be the same
    im = Image.new("RGBA", (width + left_pad + right_pad, height + top_pad + bottom_pad), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    draw.text((left_pad, top_pad), characters, (255, 255, 255), font=font)
    return characters, np.array(im)[:, :, 0].astype(np.float32) / 255., plate_mask(np.array(im).shape[0:2], radius)

def plate_mask(shape, radius):
    out = np.ones((shape[0],shape[1],1))
    out[:radius, :radius]   = 0.0
    out[-radius:, :radius]  = 0.0
    out[:radius, -radius:]  = 0.0
    out[-radius:, -radius:] = 0.0

    #this can be done entirely with numpy (using np.ogrid and a mask)
    cv2.circle(out, (radius, radius), radius, 1.0, -1)
    cv2.circle(out, (radius, shape[0] - radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, shape[0] - radius), radius, 1.0, -1)
    return out

def generate_plate(char_to_img, code = None):
    h_padding = random.uniform(0.2, 0.4) * FONT_HEIGHT
    v_padding = random.uniform(0.1, 0.3) * FONT_HEIGHT
    spacing = FONT_HEIGHT * random.uniform(-0.05, 0.05)
    radius = 1 + int(FONT_HEIGHT * 0.1 * random.random())
    if code is None:
        code = generate_code()

    text_width = np.sum([char_to_img[c].shape[1] for c in code])
    text_width += (len(code) - 1) * spacing

    out_shape = (int(FONT_HEIGHT + v_padding * 2),
                 int(text_width + h_padding * 2))

    text_mask = np.zeros(out_shape)

    x = h_padding
    y = v_padding

    for c in code:
        char_im = char_to_img[c]
        ix, iy = int(x), int(y)
        text_mask[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im
        x += char_im.shape[1] + spacing

    #case (1): simple colors, just change the number and allow for the broadcast
    #           rules to take over
    #case (2): color can be sampled from texture of the same size and shape
    #
    #Options: create the plate later (just create masks);
    #color can be changed here -----------------------v
    plate = (np.ones(out_shape)[:,:,np.newaxis] * (0.5,0.,0.) * ( 1 - text_mask)[:,:,np.newaxis] +
             np.ones(out_shape)[:,:,np.newaxis] * (0.,0.5,0.) * (text_mask)[:,:,np.newaxis])
    radius = 1 + int(FONT_HEIGHT * 0.1 * random.random())

    return code, plate, plate_mask(out_shape, radius)
    #this generates a plate

def euler_matrix(yaw, pitch, roll):
    """Construct an Euler Rotation Matrix from yaw, pitch and roll"""
    # z-y'-x'' (or z-y-x), intrinsic rotations are known as: yaw, pitch, roll

    # roll is a counter-clockwise rotation (alpha) about the x-axis
    cos_a, sin_a = np.cos(roll), np.sin(roll)
    roll_mat = np.array([[ 1,      0,      0],
                         [ 0,  cos_a, -sin_a],
                         [ 0,  sin_a,  cos_a]])

    #pitch is a counter-clockwise rotation (beta) about the y-axis
    cos_b, sin_b = np.cos(pitch), np.sin(pitch)
    pitch_mat = np.array([[ cos_b, 0, sin_b],
                          [     0, 1,     0],
                          [-sin_b, 0, cos_b]])

    #yaw is a counter-clockwise rotation (gamma) about the z-axis
    cos_g, sin_g = np.cos(yaw), np.sin(yaw)
    yaw_mat = np.array([[cos_g, -sin_g, 0],
                        [sin_g,  cos_g, 0],
                        [    0,      0, 1]])

    rotation_matrix = np.matmul(np.matmul(yaw_mat, pitch_mat), roll_mat)
    #rotate first about the x-axis, then y-axis, and finally about z-axis
    #note: if this fails might be due to numpy version
    #      (1) np.linalg.matmul(x,y)
    return rotation_matrix

def make_affine_transform(from_shape, to_shape, min_scale, max_scale,
                          scale_variation=1.0,
                          rotation_variation=1.0,
                          translation_variation=1.0):
    # original code;
    out_of_bounds = False

    from_size = np.array([[from_shape[1], from_shape[0]]]).T
    to_size   = np.array([[to_shape[1], to_shape[0]]]).T

    scale = random.uniform((min_scale + max_scale) * 0.5 -
                           (max_scale - min_scale) * 0.5 * scale_variation,
                           (min_scale + max_scale) * 0.5 +
                           (max_scale - min_scale) * 0.5 * scale_variation)

    if scale > max_scale or scale < min_scale:
        out_of_bounds = True

    roll  = random.uniform(-0.3, 0.3) * rotation_variation
    pitch = random.uniform(-0.2, 0.2) * rotation_variation
    yaw   = random.usingform(-1.2, 1.2) * rotation_variation

    M = euler_matrix(yaw=yaw, pitch=pitch, roll=roll)
    M_xy = M[:2,:2]
    h,w = from_shape

    corners = np.matrix([[-w, +w, -w, +w],
                         [-h, -h, +h, -h]]) * 0.5
    skewed_size = np.array(np.max(M_xy * corners, axis=1) - np.min(M_xy * corners, axis=1))

    scale *= np.min(to_size / skewed_size)
    trans = (np.random.random((2,1)) - 0.5) * translation_variation
    trans = ((2.0 * trans) ** 5.0) / 2.0
    if np.any(trans < -0.5) or np.any(trans > 0.5):
        out_of_bounds = True
    trans = (to_size - skewed_size * scale) * trans

    center_to = to_size / 2.
    center_from = from_size / 2.

    M = euler_matrix(yaw, pitch, roll)[:2, :2]
    M *= scale
    M = np.hstack([M, trans + center_to - M * center_from])
    #this generates an affine transformation
    return M, out_of_bounds


def generate_bg(bg_resize=True):
    files = glob.glob("/usr/share/backgrounds/*/*.jpg")
    # random.choice(files)
    # print(random.choice(files))
    found = False
    while not found:
        fname = random.choice(files)
        bg = cv2.imread(fname) / 255.#, cv2.CV_LOAD_IMAGE_GRAYSCALE) / 255.
        if (bg.shape[1] >= OUTPUT_SHAPE[1] and
            bg.shape[0] >= OUTPUT_SHAPE[0]):
            found = True


    #print(files)
    # while not found:
    #     fname = "bgs/{:08d}.jpg".format(random.randint(0, num_bg_images - 1))
    #     bg = cv2.imread(fname, cv2.CV_LOAD_IMAGE_GRAYSCALE) / 255.
    #     if (bg.shape[1] >= OUTPUT_SHAPE[1] and
    #         bg.shape[0] >= OUTPUT_SHAPE[0]):
    #         found = True
    if bg_resize:
        x_shape = np.random.randint(OUTPUT_SHAPE[1], bg.shape[1])
        y_shape = np.random.randint(OUTPUT_SHAPE[0], bg.shape[0])
        resize(image=bg, output_shape=(y_shape, x_shape), order=3)
    x = random.randint(0, bg.shape[1] - OUTPUT_SHAPE[1])
    y = random.randint(0, bg.shape[0] - OUTPUT_SHAPE[0])
    bg = bg[y:y + OUTPUT_SHAPE[0], x:x + OUTPUT_SHAPE[1]]

    return bg, fname

def generate_proposal(char_ims):
    bg, background_name = generate_bg()
    out_of_bounds = False
    code, plate, plate_mask = generate_plate(char_ims)
    # M, out_of_bounds = make_affine_transform(
    #                         from_shape=plate.shape[0:2],
    #                         to_shape=bg.shape[0:2],
    #                         min_scale=0.6,
    #                         max_scale=0.875,
    #                         rotation_variation=1.0,
    #                         scale_variation=1.5,
    #                         translation_variation=1.2)
    # print(out_of_bounds)
    # print(M.shape)
    affine = AffineTransform(rotation=-0.2, shear=0.4, scale=(0.5,0.5), translation=(20,20))
    region = np.array([ [ 0, 0,              plate.shape[1], plate.shape[1] ],
                        [ 0, plate.shape[0],              0, plate.shape[0] ],
                        [ 1, 1,                           1,             1] ])
    #print(plate.shape)
    #print("affine transformation:\n", np.matmul(affine.params[0:2,:], region))
    #
    points = np.matmul(affine.params[0:2,:], region)
    if np.min(points) < 0:
        #At least one of the points of the rectangle is out of bounds
        out_of_bounds = True
        #calculate the area of the plate out of bounds?
        # print(np.linalg.norm(points[:,0] - points[:,1]) * np.linalg.norm(points[:,0] - points[:,2]))
        # print("area     = ", np.linalg.norm(points[:,0] - points[:,1]) * np.linalg.norm(points[:,0] - points[:,2]))
        # print("original = ", np.prod(plate.shape))
        # for x in points.T:
        #     print(x)

    plate      = cv2.warpAffine(plate,      affine.params[0:2,:], (bg.shape[1], bg.shape[0]))
    plate_mask = cv2.warpAffine(plate_mask, affine.params[0:2,:], (bg.shape[1], bg.shape[0]))
    plate_mask = plate_mask[:,:,np.newaxis]

    out = plate * plate_mask + (1. - plate_mask) * bg
    out += np.random.normal(scale=0.001, size=out.shape)
    out = np.clip(out, 0., 1.)
    return out, code


def main():
    log.info(" Creating diction of character images;")
    char_ims = dict(make_character_images(FONT_HEIGHT))
    #print(char_ims.keys())
    log.info(" Characters : " + str(char_ims.keys()))
    log.info(" Quick Test of License Plate Generation (10 of them)")
    for i in range(10):
        log.info(" Licence Plate " + str(i) + " : " + generate_code())

    log.info(" Licence Plate with CODE LLDDXLLL : " + generate_code("LLDDXLLL"))
    log.info(" Licence Plate with CODE LLDDXLLL : " + generate_code("LLDDXLLL"))
    log.info(" Licence Plate with CODE LLDDQLLL : " + generate_code("LLDDQLLL"))
    log.info(" Licence Plate with CODE LLDDWLLL : " + generate_code("LLDDWLLL"))
    log.info(" Licence Plate with CODE LLDDWLLL : " + generate_code("LLDDMLLL"))
    generate_plate(char_ims)
    print(euler_matrix(1.,1.,1.))
    print(np.matmul(euler_matrix(np.pi,0,0.), np.array([1.,0.,0.])))
    print(generate_plate_alt())
    #
    print(make_affine_transform((500,200), (300,300), 0.5, 1.5))
    print(generate_bg())
    #print(generate_proposal(char_ims))
    exit(0)

if __name__ == '__main__':
    main()