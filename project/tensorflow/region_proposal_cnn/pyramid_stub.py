#!/usr/bin/env python

#code is based upon:
# http://www.pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/

import imutils
import cv2

import matplotlib.pyplot as plt

from scipy.misc import imresize
from scipy.ndimage import zoom

def pyramid(image, scale=1.5, min_size=(32,32)):
  yield image
  while True:
    w = int(image.shape[1] / scale)
    image = imutils.resize(image, width=w)
    if image.shape[0] < min_size[0] or image.shape[1] < min_size[1]:
      break
    yield image

def pyramid_scipy(image, scale=0.75, min_size=(32,32)):
  yield image
  while True:
    #Interpolation to use for re-sizing: ('nearest', 'lanczos', 'bilinear', 'bicubic' or 'cubic').
    image = imresize(image, size=scale, interp='bicubic')
    if image.shape[0] < min_size[0] or image.shape[1] < min_size[1]:
      break
    yield image

def pyramid_gaussian(image, scale=2):
  return None


def main():
  image = cv2.imread("/usr/share/backgrounds/White_flowers_by_Garuna_bor-bor.jpg")
  image[:,:,[0,2]] = image[:,:,[2,0]]
  plt.figure(figsize=(10,10))
  plt.imshow(image)
  plt.show()

  pyramids = list(pyramid(image))
  plt.figure(figsize=(10,1))
  for idx, i in enumerate(pyramids):
    plt.subplot(2,5,idx + 1)
    plt.imshow(i, interpolation='nearest')
  plt.show()

  pyramids = list(pyramid_scipy(image))
  print(len(pyramids))
  plt.figure(figsize=(10,1))
  for idx, i in enumerate(pyramids):
    plt.subplot(2,7,idx + 1)
    plt.imshow(i, interpolation='nearest')
  plt.show()



if __name__ == '__main__':
  main()
