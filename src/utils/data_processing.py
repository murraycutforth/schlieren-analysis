from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils.utils import load_cached_schlieren_images
from src.utils.paths import processed_data_dir


def main():
    if not processed_data_dir().exists():
        create_schlieren_cache()

    # Now we can load the schlieren images from the cached files
    run_to_image = load_cached_schlieren_images()

    # plot_PCA_final_frame(run_to_image)

    # Plot all frames for all runs in a 5x5 grid
    #plot_all_frames(run_to_image)

    # Plot the cumulative explained variance ratio for a full rank PCA
    X = np.stack([im.flatten() for v in run_to_image.values() for im in v], axis=0)
    X = StandardScaler().fit_transform(X)
    print(X.shape)
    pca = PCA(n_components=100)
    pca.fit(X)

    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.ylim(0, 1)
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("Cumulative explained variance ratio")
    plt.show()

    assert 0

    im_shape = next(iter(run_to_image.values())).shape[1:]
    pca = PCA(n_components=10)
    X = np.stack([v[-1].flatten() for v in run_to_image.values()], axis=0)
    X = StandardScaler().fit_transform(X)
    pca.fit(X)

    # Plot the cumulative explained variance ratio
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.ylim(0, 1)
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("Cumulative explained variance ratio")
    plt.show()

    # Plot the first 10 principal components
    fig, axs = plt.subplots(2, 5)
    for i, ax in enumerate(axs.flat):
        ax.imshow(pca.components_[i].reshape(im_shape).T)
        ax.set_title(f"PC{i}")
        ax.axis('off')

    plt.show()




    # Option: try out nonrigid registration of final frames before PCA
    # Set up data for DMD


def plot_all_frames(run_to_image):
    outdir = Path("../../data/frames")
    for timestep in range(20):
        fig, axs = plt.subplots(5, 5, figsize=(8, 12), sharex=True, sharey=True)
        for i, (run, images) in enumerate(run_to_image.items()):
            if i >= 25:
                continue
            ax = axs.flat[i]
            try:
                ax.imshow(images[timestep].T, origin="lower", cmap="gray")
            except IndexError:
                continue
            # ax.set_title(run)
            ax.axis('off')
        fig.tight_layout()
        plt.savefig(outdir / f"{timestep:02}.png")
        plt.close()


def plot_PCA_final_frame(run_to_image):
    # Now take the final frame from each of the 26 runs, and compute 2D PCA
    pca = PCA(n_components=2)
    X = np.stack([v[-1].flatten() for v in run_to_image.values()], axis=0)
    X = StandardScaler().fit_transform(X)
    pca.fit(X)
    print(f"Explained variance ratio = {pca.explained_variance_ratio_}")
    # Now plot the first two principal components
    fig, ax = plt.subplots()
    preds = pca.transform(X)
    ax.scatter(preds[:, 0], preds[:, 1])
    for i, (run, image) in enumerate(run_to_image.items()):
        ax.annotate(run, (preds[i, 0], preds[i, 1]))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA of final frame of each run")
    plt.show()


def create_schlieren_cache():
    run_to_npz_files = find_npz_files()

    # Load and plot one frame from one run
    run = next(iter(run_to_npz_files.keys()))
    plot_example_data(run_to_npz_files[run][0])

    # Now plot the first frame from each run on a 5x5 figure
    plot_first_and_last_images(run_to_npz_files)

    # Run this once to cache all images
    cache_schlieren_images(run_to_npz_files)


def cache_schlieren_images(run_to_npz_files):
    """
    Cache the schlieren images for each run in a new folder
    :return:
    """
    for run, npz_files in run_to_npz_files.items():
        frame_stack = []
        for npz_file in npz_files:
            data = np.load(npz_file)
            frame_stack.append(data["F_yz"])

        schlieren_image = np.stack(frame_stack, axis=0)
        savedir = Path(f"../../data/preprocessed")
        savedir.mkdir(exist_ok=True)
        np.savez(savedir / f"{run}.npz", schlieren_image=schlieren_image)


def find_npz_files():
    """
    Find all npz files for each run
    :return: map of run to list of npz files
    """
    npz_files = defaultdict(list)
    for npz_file in Path('../../data').glob('**/*.npz'):
        run = npz_file.parent.name
        npz_files[run].append(npz_file)

    for k, v in npz_files.items():
        npz_files[k] = sorted(v)

    return npz_files



def plot_first_and_last_images(run_to_npz_files):
    fig, axs = plt.subplots(5, 6)
    for i, k in enumerate(run_to_npz_files):
        data = np.load(run_to_npz_files[k][0])
        ax = axs.flat[i]
        ax.imshow(data["F_yz"])
        ax.set_title(k)
    fig.tight_layout()
    plt.show()
    # Now plot the last frame from each run on a 5x5 figure
    fig, axs = plt.subplots(5, 6, sharex=True, sharey=True)
    for i, k in enumerate(run_to_npz_files):
        data = np.load(run_to_npz_files[k][-1])
        ax = axs.flat[i]
        ax.imshow(data["F_yz"])
        ax.set_title(k + f" ({len(run_to_npz_files[k])} frames)")
    fig.tight_layout()
    plt.show()


def plot_example_data(npz_file):
    data = np.load(npz_file)
    ys = data['yi']
    zs = data['zi']
    Fs = data['F_yz']
    print(data)
    print(ys.shape)
    print(zs.shape)
    print(Fs.shape)
    # Plot Fs which has shape (210, 315)
    plt.figure()
    plt.imshow(Fs.T, origin="lower")
    plt.colorbar()
    plt.show()
    # Ys has shape (210, 28, 315), plot array of 28 images
    fig, axs = plt.subplots(4, 7)
    for i, ax in enumerate(axs.flat):
        ax.imshow(ys[:, i, :].T, origin="lower")
        ax.set_title(f"y{i}")
    plt.show()
    # Zs has shape (210, 28, 315), plot array of 28 images
    fig, axs = plt.subplots(4, 7)
    for i, ax in enumerate(axs.flat):
        ax.imshow(zs[:, i, :].T, origin="lower")
        ax.set_title(f"z{i}")
    plt.show()


if __name__ == '__main__':
    main()


def predict_ignition_using_final_frames(run_to_image):
    run_names = [k for k in run_to_image.keys()]

    X_last = np.stack([v[-1].flatten() for v in run_to_image.values()], axis=0)
    X_last = StandardScaler().fit_transform(X_last)

    pca = PCA(n_components=2)
    X_last_2 = pca.fit_transform(X_last)
    pca_1_threshold = 50

    ignition_indices = np.where(X_last_2[:, 0] > pca_1_threshold)[0]
    ignition_status = [True if i in ignition_indices else False for i in range(len(run_names))]
    assert sum(ignition_status) == 7
    return ignition_status
