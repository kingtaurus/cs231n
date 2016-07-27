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
import numpy

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import utils.common as common

OUTPUT_SHAPE = (64,128)
FONT_HEIGHT = 32
FONT_PATH   = "UKNumberPlate.ttf"

CHARS = common.CHARS + " "

ld_code = { 'L' : common.LETTERS, 'D' : common.DIGITS, 'S' : ' '}

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
        yield c, numpy.array(im)[:, :, 0].astype(numpy.float32) / 255.

def generate_code(lp_code = None):
    #licence descriptor
    ld = lp_code
    if lp_code is None:
        ld = list("LLDDSLLL")
    plate = ""
    for x in ld:
        plate += random.choice(ld_code[x])
    return plate
    # return "{}{}{}{} {}{}{}".format(
    #     random.choice(common.LETTERS),
    #     random.choice(common.LETTERS),
    #     random.choice(common.DIGITS),
    #     random.choice(common.DIGITS),
    #     random.choice(common.LETTERS),
    #     random.choice(common.LETTERS),
    #     random.choice(common.LETTERS))


def main():
    char_ims = dict(make_character_images(FONT_HEIGHT))
    for i in range(10):
        print(generate_code())

    exit(0)

if __name__ == '__main__':
    main()