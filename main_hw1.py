# (16-726): Project 1 starter Python code
# credit to https://inst.eecs.berkeley.edu/~cs194-26/fa18/hw/proj1/data/colorize_skel.py
# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images
import os
from pathlib import Path
import numpy as np
import skimage as sk
import skimage.io as skio
import torch

# name of the input file
imname = Path('data') / 'cathedral.jpg'

# read in the image
im = skio.imread(imname)

# convert to double (might want to do this later on to save memory)
im = sk.img_as_float(im)

# compute the height of each part (just 1/3 of total)
height = np.floor(im.shape[0] / 3.0).astype(np.int)

# separate color channels
b = im[:height]
g = im[height: 2*height]
r = im[2*height: 3*height]

# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)
def align(target, src):
    # TODO
    return target

ag = align(g, b)
ar = align(r, b)
# create a color image
im_out = np.dstack([ar, ag, b])

# save the image
os.makedirs('out', exist_ok=True)
fname = f'out/{imname.stem}.jpg'
skio.imsave(fname, im_out)

# display the image
skio.imshow(im_out)
skio.show()
