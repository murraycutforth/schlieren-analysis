import itertools

from matplotlib import pyplot as plt
from sklearn.manifold import SpectralEmbedding
import numpy as np

from src.utils.paths import output_dir
from src.utils.utils import load_cached_schlieren_images
from src.utils.data_processing import predict_ignition_using_final_frames


def main():
    run_to_image = load_cached_schlieren_images()

    # Concatenate all runs to 18 frames
    for k, v in run_to_image.items():
        run_to_image[k] = v[:18]

    seq_lens = [len(v) for v in run_to_image.values()]

    X, y = dataset_preparation(run_to_image)


    # TODO: also adjust gamma and # of neighbours for default approach

    se_approaches = \
        {
        "Time step adjacency": SpectralEmbedding(n_components=2, affinity='precomputed', random_state=0),
        "RBF": SpectralEmbedding(n_components=2, affinity='nearest_neighbors', random_state=0),
        "First frame adjacency": SpectralEmbedding(n_components=2, affinity='precomputed', random_state=0),
        }

    A_first = construct_connectivity_matrix_first(X, seq_lens)
    A_tstep = construct_connectivity_matrix_tsteps(X, seq_lens)
    fit_targets = [A_tstep, X, A_first]

    evaluate_se_approaches(se_approaches, fit_targets, X, y, seq_lens)


def evaluate_se_approaches(se_approaches, fit_targets, X, y, seq_lens):
    for i, (name, se) in enumerate(se_approaches.items()):

        X_se = se.fit_transform(fit_targets[i])

        print(f"Fit complete for {name}")

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        try:
            axs[0].imshow(se.affinity_matrix_.todense())
        except:
            axs[0].imshow(se.affinity_matrix_)

        axs[0].set_title(f"{name} affinity matrix")

        ax = axs[1]

        X_all_run_ind = np.repeat(np.arange(len(seq_lens)), seq_lens)
        cmap = plt.cm.get_cmap('jet', 18)

        for run_ind in range(len(seq_lens)):
            N = seq_lens[run_ind]
            start_ind = np.cumsum([0] + seq_lens[:-1])[run_ind]
            ignited = y[start_ind]
            X_run = X_se[start_ind:start_ind + seq_lens[run_ind], :]

            ax.plot(X_run[:18, 0], X_run[:18, 1], color="black", alpha=0.2)

            for i in range(18):
                color = cmap(i)
                marker_type = 'o' if ignited else 'x'
                ax.scatter(X_run[i, 0], X_run[i, 1], color=color, marker=marker_type)

        # Create legend and colorbar
        ax.plot([], color="b", marker="o", label="Ignited")
        ax.plot([], color="b", marker="x", label="Non-ignited")
        ax.legend()
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=18))
        sm._A = []
        fig.colorbar(sm, ax=ax, label="Time point")

        ax.set_title(f"{name} embedding")
        ax.legend()
        outdir = output_dir() / "spectral_dr"
        outdir.mkdir(exist_ok=True, parents=True)
        fig.savefig(outdir / f"se_{name}.png")
        plt.close(fig)


def construct_connectivity_matrix_tsteps(X, seq_lens):
    assert len(X) == sum(seq_lens)
    for l in seq_lens:
        assert l == 18

    start_inds = np.cumsum([0] + seq_lens[:-1])

    assert np.all(start_inds < len(X))

    A = np.zeros((len(X), len(X)))

    # Connect all start inds together
    for i, j in itertools.product(start_inds, start_inds):
        A[i, j] = 1
        A[j, i] = 1

    # For each run, connect adjacent frames together
    for start_ind, seq_len in zip(start_inds, seq_lens):
        for i in range(seq_len - 1):
            A[start_ind + i, start_ind + i + 1] = 1
            A[start_ind + i + 1, start_ind + i] = 1

    # For each run, connect to same time step in other runs
    for start_ind in start_inds:
        for i in range(18):
            for j in range(len(seq_lens)):
                A[start_ind + i, j * 18 + i] = 1
                A[j * 18 + i, start_ind + i] = 1

    # Multiply by heat kernel NN matrix
    se = SpectralEmbedding(n_components=2, affinity='nearest_neighbors', random_state=0)
    se.fit(X)
    A_euc = np.asarray(se.affinity_matrix_.todense())
    A = A * A_euc + 0.1 * A  # Add small multiple of A to avoid disconnected components

    return A



def construct_connectivity_matrix_first(X, seq_lens):
    assert len(X) == sum(seq_lens)

    start_inds = np.cumsum([0] + seq_lens[:-1])

    assert np.all(start_inds < len(X))

    A = np.zeros((len(X), len(X)))

    # Connect all start inds together
    for i, j in itertools.product(start_inds, start_inds):
        A[i, j] = 1
        A[j, i] = 1

    # For each run, connect adjacent frames together
    for start_ind, seq_len in zip(start_inds, seq_lens):
        for i in range(seq_len - 1):
            A[start_ind + i, start_ind + i + 1] = 1
            A[start_ind + i + 1, start_ind + i] = 1

        for i in range(seq_len - 2):
            A[start_ind + i, start_ind + i + 2] = 1
            A[start_ind + i + 2, start_ind + i] = 1

        for i in range(seq_len - 3):
            A[start_ind + i, start_ind + i + 3] = 1
            A[start_ind + i + 3, start_ind + i] = 1

    # Multiply by Euclidean affinity matrix
    se = SpectralEmbedding(n_components=2, affinity='nearest_neighbors', random_state=0)
    se.fit(X)
    A_euc = np.asarray(se.affinity_matrix_.todense())
    A = A * A_euc + 0.1 * A  # Add small multiple of A to avoid disconnected components

    return A


def dataset_preparation(run_to_image) -> (np.ndarray, np.ndarray):
    """Return X, y where X is a 2D data matrix (n_samples, n_features) and y is a 1D array of ignition status (n_samples,).
    """
    run_to_ignition_status = {k: v for k, v in
                              zip(run_to_image.keys(), predict_ignition_using_final_frames(run_to_image))}
    X = np.stack([im.flatten() for v in run_to_image.values() for im in v], axis=0)
    y = np.array([run_to_ignition_status[k] for k in run_to_image.keys()])
    seq_lens = [len(v) for v in run_to_image.values()]
    y = np.repeat(y, seq_lens)
    assert len(X) == len(y)
    return X, y


def embed_all_frames():
    pass


if __name__ == "__main__":
    main()
