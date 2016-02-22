"""
 Code is based from an older assignment set for CS231N.
"""

import numpy as np
import scipy.ndimage.interplolation.zoom as zoom

def resize_image(X, size=(32,32), order=3):
    """
    Generate a resized batch of images.

    Input:
    - X: (N, C, H, W)
    - size  = tuple of size_h, size_w
    - order = interpolation order

    Output:
    - An array of the shape (N,C, size_h, size_w); containing a resized
    copy of the data in X.
    """

    N, C, H, W = X.shape
    if size[0] == H and size[0] == W:
        return X.copy()

    out = np.zeros((N,C, size[0], size[1]))

    return out


def random_flips(X, p=0.5):
    """
    Generate a new batch of images where there is a random
    probability of an random x-y flip.

    Input:
    - X : (N, C, H, W)
    - p : chance of flipping an arbitrary image

    Output:
    - An array of the same shape as X (N,C,H,W); containing a
    copy of the data in X - but with 'p' examples flipped about the
    horizontal axiss.
    """

    out  = np.zeros_like(X)
    mask = np.random.choice(2, size=X.shape[0], replace=True, p=p)
    out[mask==1] = X[mask==1,:,:,::-1]
    out[mask==0] = X[mask==0,:,:,:]
    return out


def random_flips_vertical(X, p=0.5):
    """
    Generate a new batch of images where there is a random
    probability of an random x-y flip.

    Input:
    - X : (N, C, H, W)
    - p : chance of flipping an arbitrary image

    Output:
    - An array fo the same shape as X (N,C,H,W); containing a
    copy of the data in X - but with 'p' examples flipped about the
    vertical direction.
    """

    out  = np.zeros_like(X)
    mask = np.random.choice(2, size=X.shape[0], replace=True, p=p)
    out[mask==1] = X[mask==1,:,::-1,:]
    out[mask==0] = X[mask==0,:,:,:]
    return out

def random_crops(X, crop_shape):
    """
    Take random crops of images. For an input image generate a random
    crop of that image of the specified size.

    Input:
    - X: (N, C, H, W) batch of input images
    - crop_shape: Tuple of (HH,WW) to which each will be cropped.

    Output:
    -Array of shape (N, C, HH, WW)
    """

    N, C, H, W = X.shape
    HH, WW = crop_shape

    assert HH < H and WW < W

    out = np.zeros((N,C,HH,WW), dtype = X.dtype)

    # generate the associated offset for the start of crop location
    y_start = np.random.randint((H-HH), size=N)
    x_start = np.random.randint((W-WW), size=N)

    for i in range(N):
        out[i] = X[i,:,y_start[i]:y_start[i]+HH, x_start[i]:x_start[i] + WW]

    return out

def random_contrast(X, scale=(0.8,1.2)):
    """
    Randomly adjust the contrast of a batch of images.

    For an image within a batch of images, pick a number
    uniformly at random from within the 'scale' range -
    multiply all pixels by that number.

    Inputs:
    - X: (N,C,H,W) array of image data
    - scale: Tuple (low, high).

    Output:
    - Contrast rescaled array of same shape of X.
    """

    low, high = scale
    N,C = X.shape[:2]

    out = np.zeros_like(x)
    out = np.clip(np.random.uniform(low,high, size=(N,C,1,1)) * X, 0, 255)
    return out

def random_tint(X,scale=(-10,10)):
    """
    Randomly tint a batch of images.

    For an image within a batch of images, randomly pick
    a color (currently assuming RGB formatting), and additively scale
    every pixel (uniformly sampled between the range given by the scale
    parameter).

    Inputs:
    - X: (N,C,H,W) array of image data
    - scale: tuple (low,high). For each image, sample an integer in the 
      range (low,high) will be added to a random color channel.

    Output:
    - Tinted image array of shape (N, C, H, W);
    """

    low, high = scale
    N, C = X.shape[:2]

    out   = np.zeros_like(X)
    scale = np.random.randint(low=low, high=high+1, size=(N,C,1,1))
    out   = np.clip(X + scale, 0, 255)
    #clip it to a sub-range
    return out

def clip_channel(X, clip_low=[5,5,5], clip_high=[250, 250, 250] ):
    out = X.copy()
    return np.clip(out, a_min=clip_low, a_max=clip_high, out)


def fixed_crops(X, crop_shape, crop_type):
    """
    Take center or corner crops of images.
    Inputs:
    - X: Input data, of shape (N, C, H, W)
    - crop_shape: Tuple of integers (HH, WW) giving the size to which each
    image will be cropped.
    - crop_type: One of the following strings, giving the type of crop to
    compute:
    'center': Center crop
    'ul': Upper left corner
    'ur': Upper right corner
    'bl': Bottom left corner
    'br': Bottom right corner
    Returns:
    Array of cropped data of shape (N, C, HH, WW) 
    """
    N, C, H, W = X.shape
    HH, WW = crop_shape

    x0 = (W - WW) / 2
    y0 = (H - HH) / 2
    x1 = x0 + WW
    y1 = y0 + HH

    if crop_type == 'center':
        return X[:, :, y0:y1, x0:x1]
    elif crop_type == 'ul':
        return X[:, :, :HH, :WW]
    elif crop_type == 'ur':
        return X[:, :, :HH, -WW:]
    elif crop_type == 'bl':
        return X[:, :, -HH:, :WW]
    elif crop_type == 'br':
        return X[:, :, -HH:, -WW:]
    else:
        raise ValueError('Unrecognized crop type %s' % crop_type)
