from pathlib import Path

from pydmd import BOPDMD, DMD, MrDMD
from pydmd.plotter import plot_summary
import numpy as np

from src.utils.utils import load_cached_schlieren_images


def main():
    run_to_image = load_cached_schlieren_images()
    run_names = [k for k in run_to_image.keys()]
    im_shape = next(iter(run_to_image.values()))[0].shape

    # Construct the data matrix
    Xs = []
    time_points = []
    for run, ims in run_to_image.items():
        for i, im in enumerate(ims[:18]):
            Xs.append(im.flatten())
            time_points.append(i)

    X = np.stack(Xs, axis=0)
    t = np.array(time_points)
    print(X.shape, t.shape)

    # To begin with, apply Multi-resolution DMD to the data, since we have many transient effects
    # I have also found that regularisation is necessary

    sub_dmd = DMD(svd_rank=25, opt=True, tikhonov_regularization=0.1)
    mrdmd = MrDMD(sub_dmd, max_level=3, max_cycles=1)
    mrdmd.fit(X=X)

    for level, leaf, dmd in mrdmd:
        print(level, leaf, dmd.eigs)
        plot_summary(dmd)


if __name__ == '__main__':
    main()