from itertools import tee, izip, islice

import argparse
import time
import cv2

import matplotlib.pyplot as plt
import matplotlib as mpl



def window_tee(iterable, n=2):
  """ Returns a sliding window (of width n) over data from the iterable.
      s-> (s[0],..,s[n]), (s[1],..,s[n+1]), ...
  """
  iters = tee(iterable, n)
  for i in range(1,n):
    for each in iters[i:]:
      next(each, None)
  return izip(*iters)

def window_islice(iterable, n):
  """ Returns a sliding window (of width n) over data from the iterable.
      s-> [(s[0],..,s[n]), (s[1],..,s[n+1]), ...]
  """
  it = iter(iterable)
  result = tuple(islice(it, n))
  if len(result) == n:
    yield result
  for elem in it:
    result = result[1:] + (elem,)
    yield result

def sliding_window(image, window_size=(64,64), stride=(16,16)):
  for y in xrange(0, image.shape[0], stride[0]):
    for x in xrange(0, image.shape[1], stride[1]):
      yield(x, y, image[y:y+window_size[1], x:x+window_size[0],:])


def main():
  image = cv2.imread("/usr/share/backgrounds/Christmas_Lights_by_RaDu_GaLaN.jpg")
  #THIS should be equivalent to the numpy operation below:
  #cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image[:,:,[0,1,2]] = image[:,:,[2,1,0]]
  #swaps columns 0->2, 1->1, 2->0

  plt.figure(figsize=(10,10))
  plt.imshow(image)
  plt.axis('off')
  plt.show()

  windows = list(sliding_window(image, window_size=(256,256), stride=(32,32)))
  plt.figure(figsize=(10,10))
  loc_window = windows[300]
  plt.imshow(loc_window[2])
  plt.show()

  print len(windows)
  plt.figure(figsize=(10,10))
  for idx,i in enumerate(windows[0:400]):
    #print(idx)
    plt.subplot(20,20,idx+1)
    plt.imshow(i[2])
    plt.axis('off')
  plt.show()
  return 0

if __name__ == '__main__':
  main()
