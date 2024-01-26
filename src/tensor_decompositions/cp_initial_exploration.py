import numpy as np
import tensorly as tl
from matplotlib import pyplot as plt
from tensorly.decomposition import parafac

from src.utils.data_processing import predict_ignition_using_final_frames
from src.utils.utils import load_cached_schlieren_images
from src.utils.paths import output_dir


def main():
    run_to_image = load_cached_schlieren_images()
    im_shape = next(iter(run_to_image.values()))[0].shape

    ignition_status = predict_ignition_using_final_frames(run_to_image)

    # Construct the data tensor of shape (num_runs, num_features, num_time steps)

    Xs = []
    for run, ims in run_to_image.items():
        Xs.append(np.stack([im.flatten() for im in ims], axis=0)[:18])

    X = np.stack(Xs, axis=0).swapaxes(1, 2)
    X = tl.tensor(X)

    print(f"X.shape = {X.shape}")

    # Perform CP decomposition
    weights, factors = parafac(X, rank=2)

    run_factor = factors[0]  # (26, 2)
    feature_factor = factors[1]  # (66,000, 2)
    time_factor = factors[2]  # (18, 2)

    # Reconstruct a trajectory in 2D for each run (26, 18)
    trajectory_matrix_0 = np.outer(run_factor[:, 0], time_factor[:, 0])
    trajectory_matrix_1 = np.outer(run_factor[:, 1], time_factor[:, 1])
    trajectory_matrix_sum = trajectory_matrix_0 + trajectory_matrix_1

    visualise_principle_components_time_and_trial(time_factor, run_factor)

    visualise_principle_components_feature_space(feature_factor, im_shape)

    plot_trajectory(trajectory_matrix_0, trajectory_matrix_1, ignition_status)

    # Okay, so how do we reconstruct a trajectory for a new run (this would have shape (66000, 18)) ?
    # Matrix mult with feature_factor.T?


def visualise_principle_components_time_and_trial(time_factor, run_factor):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(time_factor[:, 0])
    ax.set_ylabel("First mode")
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(time_factor[:, 1])
    ax.set_ylabel("Second mode")
    ax.set_xlabel("Time")

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(run_factor[:, 0])

    ax = fig.add_subplot(2, 2, 4)
    ax.plot(run_factor[:, 1])
    ax.set_xlabel("Run")

    fig.suptitle("Canonical polyadic decomposition time and run modes")
    fig.tight_layout()
    fig.savefig(output_dir() / "cpd_time_and_run_modes.png")
    plt.show()
    plt.close()


def visualise_principle_components_feature_space(feature_factor, im_shape):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(feature_factor[:, 0].reshape(im_shape).T, cmap='gray')
    ax.set_title("First mode")
    ax.axis('off')
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(feature_factor[:, 1].reshape(im_shape).T, cmap='gray')
    ax.set_title("Second mode")
    ax.axis('off')
    fig.suptitle("Principle modes of canonical polyadic decomposition in feature space")
    fig.savefig(output_dir() / "cpd_feature_space_modes.png")
    plt.show()
    plt.close()


def plot_trajectory(trajectory_matrix_1, trajectory_matrix_2, ignition_status):
    t_vals = np.arange(trajectory_matrix_1.shape[1])
    num_runs = trajectory_matrix_1.shape[0]

    assert trajectory_matrix_2.shape == trajectory_matrix_1.shape
    assert len(ignition_status) == num_runs

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"CPD trajectory plot")

    ax = axs[0]

    ax.set_xlabel("First mode")
    ax.set_ylabel("Second mode")
    cmap = plt.get_cmap('jet', len(t_vals))

    for run_ind in range(num_runs):
        ignited = ignition_status[run_ind]

        ax.plot(trajectory_matrix_1[run_ind, :], trajectory_matrix_2[run_ind, :], color="black", alpha=0.2)

        for i in range(len(t_vals)):
            color = cmap(i)
            marker_type = 'o' if ignited else 'x'
            ax.scatter(trajectory_matrix_1[run_ind, i], trajectory_matrix_2[run_ind, i], color=color, marker=marker_type)

    # Create legend and colorbar
    handles = [plt.Line2D([0], [0], color='black', marker='o', linestyle='None'),
               plt.Line2D([0], [0], color='black', marker='x', linestyle='None')]
    labels = ['Ignited', 'Non-ignited']
    ax.legend(handles, labels)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max(t_vals)))
    sm._A = []
    fig.colorbar(sm, ax=ax, label="Time point")

    # On the second axis plot a zoomed version looking at just the first five time points

    ax = axs[1]

    ax.set_xlabel("First mode")
    ax.set_ylabel("Second mode")
    cmap = plt.get_cmap('jet', 5)

    for run_ind in range(num_runs):
        ignited = ignition_status[run_ind]

        ax.plot(trajectory_matrix_1[run_ind, :5], trajectory_matrix_2[run_ind, :5], color="black", alpha=0.2)

        for i in range(5):
            color = cmap(i)
            marker_type = 'o' if ignited else 'x'
            ax.scatter(trajectory_matrix_1[run_ind, i], trajectory_matrix_2[run_ind, i], color=color, marker=marker_type)

    # Create legend and colorbar
    handles = [plt.Line2D([0], [0], color='black', marker='o', linestyle='None'),
               plt.Line2D([0], [0], color='black', marker='x', linestyle='None')]
    labels = ['Ignited', 'Non-ignited']
    ax.legend(handles, labels)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=5))
    sm._A = []
    fig.colorbar(sm, ax=ax, label="Time point")

    fig.tight_layout()
    plt.show()
    fig.savefig(output_dir() / "cpd_trajectory_plot.png")
    plt.close()


if __name__ == '__main__':
    main()