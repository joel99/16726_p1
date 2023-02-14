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
import skimage.transform as skt
import torch
import argparse

use_pytorch = False
use_pytorch = True
lib_func = torch if use_pytorch else np
BASE_SWEEP = 30 # used for cathedral i.e. when sweep isn't specified for single-scale
BASE_SWEEP_RANGE = torch.tensor([-BASE_SWEEP, BASE_SWEEP]) if use_pytorch else np.array([-BASE_SWEEP, BASE_SWEEP])
def score_similarity(im1: torch.Tensor, im2: torch.Tensor, mode='ssd') -> float:
    # Return higher is better
    # SSD for starters
    if mode == 'ssd':
        return -lib_func.sum((im1 - im2) ** 2)
    elif mode == 'ncc':
        return lib_func.sum(im1 * im2) / (lib_func.sqrt(lib_func.sum(im1 ** 2)) * lib_func.sqrt(lib_func.sum(im2 ** 2)))
    else:
        raise NotImplementedError


def main(input_path, tag, metric, crop_pct, sweep):
    # name of the input file
    imname = Path(input_path)

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

    def align(
        src: np.ndarray,
        target: np.ndarray,
        sweep_x=BASE_SWEEP_RANGE,
        sweep_y=BASE_SWEEP_RANGE,
        crop_pct=crop_pct
    ):
        # print(src.shape, target.shape, sweep_x, sweep_y, crop_pct)
        if use_pytorch:
            src = torch.tensor(src)
            target = torch.tensor(target)
        scores = lib_func.zeros((sweep_x[1] - sweep_x[0], sweep_y[1] - sweep_y[0]))
        for i, offset_x in enumerate(range(sweep_x[0], sweep_x[1])):
            for j, offset_y in enumerate(range(sweep_y[0], sweep_y[1])):
                shift_src = lib_func.roll(src, (offset_x, offset_y), (0, 1))
                # crop both shift_src and target for comparison
                # if shift is positive, then the start is invalid
                # if shift is negative, then the end is invalid
                def get_valid_crop(src: torch.Tensor, offset, axis=0):
                    indices = lib_func.arange(src.shape[axis])
                    valid_mask = lib_func.ones(indices.shape, dtype=bool)
                    valid_mask[:max(0, offset)] = False
                    if offset < 0:
                        valid_mask[offset:] = False
                    indices = indices[valid_mask]
                    if use_pytorch:
                        return torch.index_select(src, axis, indices)
                    else:
                        return np.take(src, indices, axis=axis)
                def center_crop(src: torch.Tensor, pct):
                    divide_func = torch.div if use_pytorch else np.divide
                    center_x = divide_func(src.shape[0], 2)
                    center_y = divide_func(src.shape[1], 2)
                    span_x = (src.shape[0] * pct / 2)
                    span_y = (src.shape[1] * pct / 2)
                    return src[int(center_x - span_x):int(center_x + span_x), int(center_y - span_y):int(center_y + span_y)]
                shift_src_valid = get_valid_crop(shift_src, offset_x, axis=0)
                shift_src_valid = get_valid_crop(shift_src_valid, offset_y, axis=1)
                target_valid = get_valid_crop(target, offset_x, axis=0) # Note target hasn't shifted but the valid region of comparison is identically affected
                target_valid = get_valid_crop(target_valid, offset_y, axis=1)
                # * Note, metrics won't work well if we compare variable regions depending on roll. Take a fixed (centered) crop
                shift_src_valid = center_crop(shift_src_valid, crop_pct)
                target_valid = center_crop(target_valid, crop_pct)
                score = score_similarity(shift_src_valid, target_valid, mode=metric)
                scores[i, j] = score
        best_index = lib_func.argmax(scores)
        x_index = torch.div(best_index, scores.shape[0], rounding_mode='trunc') if use_pytorch else best_index // scores.shape[0]
        roll_x = range(sweep_x[0], sweep_x[1])[x_index]
        roll_y = range(sweep_y[0], sweep_y[1])[best_index % scores.shape[0]]
        shift_src = lib_func.roll(src, (roll_x, roll_y), (0, 1))
        # print(f'{roll_x} from {sweep_x}, {roll_y} from {sweep_y}')
        # print(roll_x, roll_y)
        return shift_src, (roll_x, roll_y)

    def align_multiscale(src: np.ndarray, target: np.ndarray, scale_res_min=100, iter_factor=2, sweep_range=[-sweep, sweep]):
        assert src.shape == target.shape # TODO - support different shapes
        if use_pytorch:
            sweep_range = torch.tensor(sweep_range)
        else:
            sweep_range = np.array(sweep_range)
        scale_factor = 1
        scale_shape = max(src.shape)
        while scale_shape > scale_res_min:
            scale_shape = scale_shape // iter_factor
            scale_factor = scale_factor * iter_factor
        roll_x, roll_y = 0, 0
        while scale_factor >= 1:
            # print(f'Aligning at scale {scale_factor}')
            roll_x *= iter_factor
            roll_y *= iter_factor
            src_scale = skt.rescale(src, 1 / scale_factor, anti_aliasing=True)
            target_scale = skt.rescale(target, 1 / scale_factor, anti_aliasing=True)
            # print(f'scale: {scale_factor} roll center: {roll_x}, {roll_y}')
            _, (roll_x, roll_y) = align(src_scale, target_scale, sweep_x=sweep_range + roll_x, sweep_y=sweep_range + roll_y)
            # print(f'scale: {scale_factor} roll after: {roll_x}, {roll_y}')
            scale_factor = int(scale_factor / iter_factor)
            # while the scaling will have rounding errors, the extra sweeping margin should be more than enough to make up for it.
        if use_pytorch:
            src = torch.tensor(src)
        shift_src = lib_func.roll(src, (roll_x, roll_y), (0, 1))
        return shift_src, (roll_x, roll_y)

    ag, rolls_g = align_multiscale(g, b, scale_res_min=100, iter_factor=2, sweep_range=[-sweep, sweep])
    ar, rolls_r = align_multiscale(r, b, scale_res_min=100, iter_factor=2, sweep_range=[-sweep, sweep])
    # ag, _ = align(g, b, crop_pct=crop_pct)
    # ar, _ = align(r, b, crop_pct=crop_pct)
    if use_pytorch:
        im_out = torch.stack([ar, ag, torch.tensor(b)], -1)
        im_out = (im_out).numpy()
    else:
        im_out = np.dstack([ar, ag, b])
    im_out = (im_out * 255).astype(np.uint8)
    # save the image
    out_root = Path('out') / tag / metric
    os.makedirs(out_root, exist_ok=True)
    fname = out_root / f'{imname.stem}_g_{rolls_g}_r_{rolls_r}.jpg'
    skio.imsave(fname, im_out)

    # display the image
    # skio.imshow(im_out)
    # skio.show()

# Notes
# SSD is trash, cathedral doesn't work on it even just with a slightly larger than 4x4 search range; too brittle

if __name__ == '__main__':
    # parse for input file
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', '-i', type=str, default='data/cathedral.jpg')
    parser.add_argument('--tag', '-t', type=str, default='numpy_base')
    parser.add_argument('--metric', '-m', type=str, default='ssd') # ssd, ncc
    parser.add_argument('--crop-pct', '-c', type=float, default=0.5, help='pct of image to compare')
    parser.add_argument('--sweep', '-s', type=int, default=4, help='brute force search range')
    # parser.add_argument('--sweep', '-s', type=int, default=8, help='brute force search range')
    args = parser.parse_args()

    args = parser.parse_args()
    main(**vars(args))
