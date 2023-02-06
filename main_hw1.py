#%%
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
height = np.floor(im.shape[0] / 3.0).astype(int)

# separate color channels
b = im[:height]
g = im[height: 2*height]
r = im[2*height: 3*height]


def score_similarity(im1, im2, mode='ssd') -> float:
    # Return higher is better
    # SSD for starters
    if mode == 'ssd':
        return -torch.sum((im1 - im2) ** 2)
    elif mode == 'ncc':
        return torch.sum(im1 * im2) / (torch.sqrt(torch.sum(im1 ** 2)) * torch.sqrt(torch.sum(im2 ** 2)))
    else:
        raise NotImplementedError

# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)

# TODO - get this working
# * Piazza says crop more - let's do that
# * Piazza says SSD should work reasonably

# SWEEP = 15
SWEEP = 30 # Sufficient for cathedral
# SWEEP = 45
# SWEEP = 90
SIMILARITY_METRIC = 'ssd'
# SIMILARITY_METRIC = 'ncc'
CROP_PCT = 0.5


def align(src: np.ndarray, target: np.ndarray, sweep_range=[-SWEEP, SWEEP]):
    src = torch.tensor(src)
    target = torch.tensor(target)
    scores = torch.zeros((sweep_range[1] - sweep_range[0], sweep_range[1] - sweep_range[0]))
    for i, offset_x in enumerate(range(sweep_range[0], sweep_range[1])):
        for j, offset_y in enumerate(range(sweep_range[0], sweep_range[1])):
            shift_src = torch.roll(src, (offset_x, offset_y), dims=(0, 1))
            # crop both shift_src and target for comparison
            # if shift is positive, then the start is invalid
            # if shift is negative, then the end is invalid
            def get_valid_crop(src: torch.Tensor, offset, axis=0):
                indices = torch.arange(src.size(axis))
                valid_mask = torch.ones(indices.size(), dtype=torch.bool)
                valid_mask[:max(0, offset)] = False
                if offset < 0:
                    valid_mask[offset:] = False
                indices = indices[valid_mask]
                return torch.index_select(src, axis, indices)
            def center_crop(src: torch.Tensor, pct):
                center_x = torch.div(src.shape[0], 2)
                center_y = torch.div(src.shape[1], 2)
                span_x = (src.shape[0] * pct / 2)
                span_y = (src.shape[1] * pct / 2)
                return src[int(center_x - span_x):int(center_x + span_x), int(center_y - span_y):int(center_y + span_y)]
            shift_src_valid = get_valid_crop(shift_src, offset_x, axis=0)
            shift_src_valid = get_valid_crop(shift_src_valid, offset_y, axis=1)
            target_valid = get_valid_crop(target, offset_x, axis=0) # Note target hasn't shifted but the valid region of comparison is identically affected
            target_valid = get_valid_crop(target_valid, offset_y, axis=1)
            # * Note, metrics won't work well if we compare variable regions depending on roll. Take a fixed (centered) crop
            shift_src_valid = center_crop(shift_src_valid, CROP_PCT)
            target_valid = center_crop(target_valid, CROP_PCT)
            score = score_similarity(shift_src_valid, target_valid, mode=SIMILARITY_METRIC)
            scores[i, j] = score
    best_index = torch.argmax(scores)
    roll_x = range(sweep_range[0], sweep_range[1])[torch.div(best_index, scores.shape[0], rounding_mode='trunc')]
    roll_y = range(sweep_range[0], sweep_range[1])[best_index % scores.shape[0]]
    shift_src = torch.roll(src, (roll_x, roll_y), dims=(0, 1))
    return shift_src, scores

#%%
# g_roll = np.roll(g, (100, 0 ), axis=(0, 1))
# g_roll = np.roll(g, (100, 100), axis=(0, 1))
# skio.imshow(g_roll)
# skio.imshow(g)
# Roll works as expected in mind
# Currently I don't know if my cropping is bad, my final selection is bad, or what
# Maybe let's manually inspect the anticipated roll
#%%
# ag = torch.roll(torch.tensor(g), (5, 0), dims=(0, 1))
# ag = torch.roll(torch.tensor(g), (0, 0), dims=(0, 1))
# ag = torch.roll(torch.tensor(g), (-25, 0), dims=(0, 1))
# ? Why am I perceiving that rolling negative moves green down? Rolling negative moves green plate up
# * DWAI basically. Intuition about color blocks is wrong.

ag, _ = align(g, b)
# ar = torch.tensor(r).fill_(0)
ar, _ = align(r, b)
# im_out = torch.stack([ar, ag, torch.tensor(b).fill_(0)], -1)
im_out = torch.stack([ar, ag, torch.tensor(b)], -1)
im_out = (im_out * 255).numpy().astype(np.uint8)
# im_out = np.dstack([ar, ag, b])

# save the image
os.makedirs('out', exist_ok=True)
fname = f'out/{imname.stem}.jpg'
skio.imsave(fname, im_out)

# display the image
skio.imshow(im_out)
skio.show()
