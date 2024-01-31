import numpy as np
import tensorly as tl
from matplotlib import pyplot as plt
from tensorly.decomposition import parafac, tucker, non_negative_parafac, non_negative_tucker
from tensorly.tenalg import mode_dot

from src.utils.data_processing import predict_ignition_using_final_frames
from src.utils.utils import load_cached_schlieren_images
from src.utils.paths import output_dir


def main():
    run_to_image = load_cached_schlieren_images()
    im_shape = next(iter(run_to_image.values()))[0].shape
    ignition_status = predict_ignition_using_final_frames(run_to_image)

    # Construct the data tensor of shape (num_runs, num_features, num_time steps)
    X = construct_data_tensor(run_to_image)

    # Perform Tucker and N Tucker decompositions

    for fn, name in [(tucker, "tucker"), (non_negative_tucker, "non-negative tucker")]:
        core, factors = fn(X, rank=2)

        run_factor = factors[0]
        feature_factor = factors[1]
        time_factor = factors[2]

        # Reconstruct a trajectory in 2D for each run (26, 18)
        trajectory_tensor = mode_dot(mode_dot(core, run_factor, mode=0), time_factor, mode=2)
        trajectory_matrix_0 = trajectory_tensor[:, 0, :]
        trajectory_matrix_1 = trajectory_tensor[:, 1, :]

        visualise_principle_components(time_factor, run_factor, feature_factor, im_shape, name)
        plot_trajectory(trajectory_matrix_0, trajectory_matrix_1, ignition_status, name)


    # Perform CP and NCP decompositions

    for fn, name in [(parafac, "cp"), (non_negative_parafac, "non-negative cp")]:
        weights, factors = fn(X, rank=2)

        print(f"Weights: {weights}")

        run_factor = factors[0]  # (26, 2)
        feature_factor = factors[1]  # (66,000, 2)
        time_factor = factors[2]  # (18, 2)

        # Reconstruct a trajectory in 2D for each run (26, 18)
        trajectory_matrix_0 = np.outer(run_factor[:, 0], time_factor[:, 0])
        trajectory_matrix_1 = np.outer(run_factor[:, 1], time_factor[:, 1])

        visualise_principle_components(time_factor, run_factor, feature_factor, im_shape, name)
        plot_trajectory(trajectory_matrix_0, trajectory_matrix_1, ignition_status, name)



def construct_data_tensor(run_to_image):
    Xs = []
    for run, ims in run_to_image.items():
        Xs.append(np.stack([im.flatten() for im in ims], axis=0)[:18])
    X = np.stack(Xs, axis=0).swapaxes(1, 2)
    X = tl.tensor(X)
    print(f"X.shape = {X.shape}")
    return X


def visualise_principle_components(time_factor, run_factor, feature_factor, im_shape, name):
    fig = plt.figure(figsize=(12, 8))

    ax = fig.add_subplot(2, 3, 1)
    ax.plot(time_factor[:, 0])
    ax.set_ylabel("First mode")

    ax = fig.add_subplot(2, 3, 4)
    ax.plot(time_factor[:, 1])
    ax.set_ylabel("Second mode")
    ax.set_xlabel("Time")

    ax = fig.add_subplot(2, 3, 2)
    ax.plot(run_factor[:, 0])

    ax = fig.add_subplot(2, 3, 5)
    ax.plot(run_factor[:, 1])
    ax.set_xlabel("Run")

    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(feature_factor[:, 0].reshape(im_shape).T, cmap='gray')
    ax.axis('off')

    ax = fig.add_subplot(2, 3, 6)
    ax.imshow(feature_factor[:, 1].reshape(im_shape).T, cmap='gray')
    ax.axis('off')
    ax.set_xlabel("Feature (reshaped to image)")

    fig.tight_layout()

    (output_dir() / "tensors").mkdir(exist_ok=True, parents=True)
    fig.savefig(output_dir() / "tensors" / f"{name}_time_and_run_modes.png")

    plt.close()


def plot_trajectory(trajectory_matrix_1, trajectory_matrix_2, ignition_status, name):
    t_vals = np.arange(trajectory_matrix_1.shape[1])
    num_runs = trajectory_matrix_1.shape[0]

    assert trajectory_matrix_2.shape == trajectory_matrix_1.shape
    assert len(ignition_status) == num_runs

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"{name} trajectory plot")

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
    fig.savefig(output_dir() / "tensors" / f"{name}_trajectory_plot.png")
    plt.close()


if __name__ == '__main__':
    main()