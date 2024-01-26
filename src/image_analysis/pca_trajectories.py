from typing import Callable

import skimage.measure
from skimage import filters
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils.data_processing import predict_ignition_using_final_frames
from src.utils.utils import load_cached_schlieren_images


def main():
    run_to_image = load_cached_schlieren_images()
    run_names = [k for k in run_to_image.keys()]
    im_shape = next(iter(run_to_image.values())).shape[1:]

    # Create data matrix for all frames, for the first frames only, and for the last frames only
    X_first = np.stack([v[0].flatten() for v in run_to_image.values()], axis=0)
    X_first = StandardScaler().fit_transform(X_first)

    print(X_first.min())
    print(X_first.max())
    print(f"X_first column statistics: mean = {np.mean(X_first, axis=0)}, std = {np.std(X_first, axis=0)}")
    assert 0

    X_last = np.stack([v[-1].flatten() for v in run_to_image.values()], axis=0)
    X_last = StandardScaler().fit_transform(X_last)
    X_all = np.stack([im.flatten() for v in run_to_image.values() for im in v], axis=0)
    X_all = StandardScaler().fit_transform(X_all)

    ignition_status = predict_ignition_using_final_frames(run_to_image)

    # Plot all three PCA plots
    #plot_PCA_comparison(X_all_2, X_all_2_pc, X_first_2, X_first_2_pc, X_last_2, X_last_2_pc, im_shape)

    # Plot trajectories through different 2D PCA spaces
    #plot_pca_trajectories(X_all, X_first, X_last, ignition_status, run_to_image)

    # IDEA - PCA/POD is designed to maximise the variance of the data in each mode
    # We are interested in trajectories of time series through a new 2D subspace
    # Can we find a subspace which optimally splits the ignition and non-ignition trajectories?

    # 2nd IDEA - Latent Dirichlet Allocation on discretised data

    # Okay so now I want to play around with different downscalings and smoothings of the data before showing the PCA trajectory
    # Only consider the full dataset

    investigate_image_filters_on_pca_trajectory(run_to_image, ignition_status)


def investigate_image_filters_on_pca_trajectory(run_to_image, ignition_status):

    def plot_trajectory_given_transform(fn: Callable, fn_name: str):
        X_all_transformed = np.stack([fn(im).flatten() for v in run_to_image.values() for im in v], axis=0)
        X_all_transformed = StandardScaler().fit_transform(X_all_transformed)
        plot_transformed_images(fn, run_to_image, fn_name)
        plot_pca_trajectory(X_all_transformed, ignition_status, run_to_image, fn_name)

    def image_quantisation(im):
        return np.round((im / im.max()) * 5)

    plot_trajectory_given_transform(image_quantisation, "Image quantisation - 5 levels")

    def otsu_threshold(im):
        return im > filters.threshold_otsu(im)

    plot_trajectory_given_transform(otsu_threshold, "Otsu threshold")

    def constant_threshold(im):
        return im > 1.0

    plot_trajectory_given_transform(constant_threshold, "Constant threshold = 1")

    def local_entropy(im):
        return filters.rank.entropy(im / (1 + im.max() - im.min()), disk(5))

    plot_trajectory_given_transform(local_entropy, "Local entropy")

    def butterworth_high_pass(im):
        return filters.butterworth(im, cutoff_frequency_ratio=0.1, order=3, high_pass=True, npad=4)

    plot_trajectory_given_transform(butterworth_high_pass, "Butterworth high pass")
    plot_trajectory_given_transform(lambda x: x, "Original")

    def smooth_image_2sigma(im):
        return filters.gaussian(im, sigma=2, truncate=3.0)

    plot_trajectory_given_transform(smooth_image_2sigma, "Smoothed - 2 $\sigma$")

    def smooth_image_10sigma(im):
        im = filters.gaussian(im, sigma=10, truncate=3.0)
        return skimage.measure.block_reduce(im, (2, 2), np.mean)

    plot_trajectory_given_transform(smooth_image_10sigma, "Smoothed and downsampled - 10 $\sigma$")

    def smooth_image_20sigma(im):
        im = filters.gaussian(im, sigma=20, truncate=3.0)
        return skimage.measure.block_reduce(im, (4, 4), np.mean)

    plot_trajectory_given_transform(smooth_image_20sigma, "Smoothed and downsampled - 20 $\sigma$")

    def sobel_filter(im):
        return filters.sobel(im)

    plot_trajectory_given_transform(sobel_filter, "Sobel filter")


def plot_transformed_images(fn, run_to_image, fn_name):
    # Plot the transformed images - just show plots from a non-ignition and an ignition run

    run1 = "schlieren_COARSE_sim_26"
    run2 = "schlieren_COARSE_sim_39"

    for run in [run1, run2]:

        ims = run_to_image[run]
        ims = list(map(fn, ims))

        fig, axs = plt.subplots(3, 6, figsize=(12, 10))

        for i, ax in enumerate(axs.flat):
            try:
                ax.imshow(ims[i].T, cmap='gray')
            except IndexError:
                pass
            ax.axis('off')

        fig.tight_layout()
        fig.savefig(f"output/transformed_images_{fn_name}_{run}.png")
        fig.close()


def plot_pca_trajectory(X_all, ignition_status, run_to_image, fn_name):
    # Plot trajectories of frames, coloured according to their time point
    num_time_points = [len(v) for v in run_to_image.values()]
    X_all_run_ind = np.repeat(np.arange(len(run_to_image)), num_time_points)
    X_all_iginition = np.repeat(ignition_status, num_time_points)

    pca = PCA(n_components=2)
    pca.fit(X_all)
    X_all_2 = pca.transform(X_all)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"PCA trajectory (image preprocessing: {fn_name})")

    ax = axs[0]

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    cmap = plt.get_cmap('jet', max(num_time_points))

    for run_ind in range(len(run_to_image)):
        N = num_time_points[run_ind]
        ignited = ignition_status[run_ind]
        X = X_all_2[X_all_run_ind == run_ind, :]
        assert len(X) == N

        ax.plot(X[:, 0], X[:, 1], color="black", alpha=0.2)

        for i in range(len(X)):
            color = cmap(i)
            marker_type = 'o' if ignited else 'x'
            ax.scatter(X[i, 0], X[i, 1], color=color, marker=marker_type)

    # Create legend and colorbar
    handles = [plt.Line2D([0], [0], color='black', marker='o', linestyle='None'),
               plt.Line2D([0], [0], color='black', marker='x', linestyle='None')]
    labels = ['Ignited', 'Non-ignited']
    ax.legend(handles, labels)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max(num_time_points)))
    sm._A = []
    fig.colorbar(sm, ax=ax, label="Time point")

    # On the second axis plot a zoomed version looking at just the first five time points

    ax = axs[1]

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    cmap = plt.get_cmap('jet', 5)

    for run_ind in range(len(run_to_image)):
        N = num_time_points[run_ind]
        ignited = ignition_status[run_ind]
        X = X_all_2[X_all_run_ind == run_ind, :]
        assert len(X) == N

        ax.plot(X[:5, 0], X[:5, 1], color="black", alpha=0.2)

        for i in range(5):
            color = cmap(i)
            marker_type = 'o' if ignited else 'x'
            ax.scatter(X[i, 0], X[i, 1], color=color, marker=marker_type)

    # Create legend and colorbar
    handles = [plt.Line2D([0], [0], color='black', marker='o', linestyle='None'),
               plt.Line2D([0], [0], color='black', marker='x', linestyle='None')]
    labels = ['Ignited', 'Non-ignited']
    ax.legend(handles, labels)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=5))
    sm._A = []
    fig.colorbar(sm, ax=ax, label="Time point")

    fig.tight_layout()
    fig.savefig(f"output/pca_trajectory_{fn_name}.png")
    fig.close()


def plot_pca_trajectories(X_all, X_first, X_last, ignition_status, run_to_image):
    # Plot trajectories of frames, coloured according to their time point
    num_time_points = [len(v) for v in run_to_image.values()]
    X_all_run_ind = np.repeat(np.arange(len(run_to_image)), num_time_points)
    X_all_iginition = np.repeat(ignition_status, num_time_points)
    pca = PCA(n_components=2)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].set_title("First time point only")
    axs[0].set_xlabel("PC1")
    axs[0].set_ylabel("PC2")
    pca.fit(X_first)
    X_all_2 = pca.transform(X_all)
    cmap = plt.get_cmap('jet', max(num_time_points))
    for run_ind in range(len(run_to_image)):
        N = num_time_points[run_ind]
        ignited = ignition_status[run_ind]
        X = X_all_2[X_all_run_ind == run_ind, :]
        assert len(X) == N

        axs[0].plot(X[:, 0], X[:, 1], color="black", alpha=0.2)

        for i in range(len(X)):
            color = cmap(i)
            marker_type = 'o' if ignited else 'x'
            axs[0].scatter(X[i, 0], X[i, 1], color=color, marker=marker_type)
    # Create legend and colorbar
    handles = [plt.Line2D([0], [0], color='black', marker='o', linestyle='None'),
               plt.Line2D([0], [0], color='black', marker='x', linestyle='None')]
    labels = ['Ignited', 'Non-ignited']
    axs[0].legend(handles, labels)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max(num_time_points)))
    sm._A = []
    fig.colorbar(sm, ax=axs[0], label="Time point")
    axs[1].set_title("Last time point only")
    axs[1].set_xlabel("PC1")
    axs[1].set_ylabel("PC2")
    pca.fit(X_last)
    X_all_2 = pca.transform(X_all)
    cmap = plt.get_cmap('jet', max(num_time_points))
    for run_ind in range(len(run_to_image)):
        N = num_time_points[run_ind]
        ignited = ignition_status[run_ind]
        X = X_all_2[X_all_run_ind == run_ind, :]
        assert len(X) == N

        axs[1].plot(X[:, 0], X[:, 1], color="black", alpha=0.2)

        for i in range(len(X)):
            color = cmap(i)
            marker_type = 'o' if ignited else 'x'
            axs[1].scatter(X[i, 0], X[i, 1], color=color, marker=marker_type)
    # Create legend and colorbar
    handles = [plt.Line2D([0], [0], color='black', marker='o', linestyle='None'),
               plt.Line2D([0], [0], color='black', marker='x', linestyle='None')]
    labels = ['Ignited', 'Non-ignited']
    axs[1].legend(handles, labels)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max(num_time_points)))
    sm._A = []
    fig.colorbar(sm, ax=axs[1], label="Time point")
    axs[2].set_title("All time points")
    axs[2].set_xlabel("PC1")
    axs[2].set_ylabel("PC2")
    pca.fit(X_all)
    X_all_2 = pca.transform(X_all)
    cmap = plt.get_cmap('jet', max(num_time_points))
    for run_ind in range(len(run_to_image)):
        N = num_time_points[run_ind]
        ignited = ignition_status[run_ind]
        X = X_all_2[X_all_run_ind == run_ind, :]
        assert len(X) == N

        axs[2].plot(X[:, 0], X[:, 1], color="black", alpha=0.2)

        for i in range(len(X)):
            color = cmap(i)
            marker_type = 'o' if ignited else 'x'
            axs[2].scatter(X[i, 0], X[i, 1], color=color, marker=marker_type)
    # Create legend and colorbar
    handles = [plt.Line2D([0], [0], color='black', marker='o', linestyle='None'),
               plt.Line2D([0], [0], color='black', marker='x', linestyle='None')]
    labels = ['Ignited', 'Non-ignited']
    axs[2].legend(handles, labels)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max(num_time_points)))
    sm._A = []
    fig.colorbar(sm, ax=axs[2], label="Time point")
    fig.tight_layout()
    plt.show()


def plot_PCA_comparison(X_all_2, X_all_2_pc, X_first_2, X_first_2_pc, X_last_2, X_last_2_pc, im_shape):
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    axs = axs.T
    axs[0, 0].scatter(X_first_2[:, 0], X_first_2[:, 1])
    axs[0, 0].set_xlabel("PC1")
    axs[0, 0].set_ylabel("PC2")
    axs[0, 0].set_title("First time point only")
    axs[0, 1].scatter(X_last_2[:, 0], X_last_2[:, 1])
    axs[0, 1].set_xlabel("PC1")
    axs[0, 1].set_ylabel("PC2")
    axs[0, 1].set_title("Final time point only")
    axs[0, 2].scatter(X_all_2[:, 0], X_all_2[:, 1])
    axs[0, 2].set_xlabel("PC1")
    axs[0, 2].set_ylabel("PC2")
    axs[0, 2].set_title("All time points")

    axs[1, 0].imshow(X_first_2_pc[0].reshape(im_shape).T)
    axs[1, 0].set_title(f"First mode")
    axs[1, 0].axis('off')
    axs[1, 1].imshow(X_last_2_pc[0].reshape(im_shape).T)
    axs[1, 1].axis('off')
    axs[1, 2].imshow(X_all_2_pc[0].reshape(im_shape).T)
    axs[1, 2].axis('off')

    axs[2, 0].imshow(X_first_2_pc[1].reshape(im_shape).T)
    axs[2, 0].set_title(f"Second mode")
    axs[2, 0].axis('off')
    axs[2, 1].imshow(X_last_2_pc[1].reshape(im_shape).T)
    axs[2, 1].axis('off')
    axs[2, 2].imshow(X_all_2_pc[1].reshape(im_shape).T)
    axs[2, 2].axis('off')

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()