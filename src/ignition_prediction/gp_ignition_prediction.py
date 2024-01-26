from collections import defaultdict
from pathlib import Path

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import train_test_split
from skimage import filters
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils.data_processing import predict_ignition_using_final_frames
from utils import load_cached_schlieren_images, construct_full_data_matrix


def main():
    run_to_image = load_cached_schlieren_images()
    run_names = [k for k in run_to_image.keys()]

    # Predict ignition, which we will use for a stratified split of data into train and test
    X_last = np.stack([v[-1].flatten() for v in run_to_image.values()], axis=0)
    X_last = StandardScaler().fit_transform(X_last)
    ignition_status = predict_ignition_using_final_frames(X_last, run_names)
    run_to_ignition = {k: v for k, v in zip(run_names, ignition_status)}

    train_runs, test_runs, train_ignition, test_ignition = train_test_split(run_names, ignition_status, test_size=0.2, random_state=42, stratify=ignition_status)
    train_ignition = [run_to_ignition[k] for k in run_to_image if k in train_runs]
    test_ignition = [run_to_ignition[k] for k in run_to_image if k in test_runs]

    print(f"Train runs = {train_runs}")
    print(f"Test runs = {test_runs}")
    print(f"Train ignition = {train_ignition}")
    print(f"Test ignition = {test_ignition}")

    # Construct and normalise the data matrix for the training and test sets
    X_train = np.stack([im.flatten() for k, v in run_to_image.items() for im in v if k in train_runs], axis=0)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(np.stack([im.flatten() for k, v in run_to_image.items() for im in v if k in test_runs], axis=0))

    y_train = np.repeat(np.array(train_ignition, dtype=int), [len(run_to_image[k]) for k in run_to_image if k in train_runs])
    y_test = np.repeat(np.array(test_ignition, dtype=int), [len(run_to_image[k]) for k in run_to_image if k in test_runs])
    train_run_names = np.repeat(train_runs, [len(run_to_image[k]) for k in run_to_image if k in train_runs])
    train_image_timepoint = np.concatenate([np.arange(len(run_to_image[k])) for k in run_to_image if k in train_runs])
    test_run_names = np.repeat(test_runs, [len(run_to_image[k]) for k in run_to_image if k in test_runs])
    test_image_timepoint = np.concatenate([np.arange(len(run_to_image[k])) for k in run_to_image if k in test_runs])

    # Reduce dimensionality using PCA, and re-whiten data
    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(pca.transform(X_test))

    print(f"X_train.shape = {X_train.shape}")
    print(f"X_test.shape = {X_test.shape}")
    print(f"y_train.shape = {y_train.shape}")
    print(f"y_test.shape = {y_test.shape}")

    assert len(train_run_names) == len(X_train)
    assert len(y_train) == len(X_train)

    # Create GP kernel
    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)) + 0.1 * RBF(length_scale=0.1, length_scale_bounds=(1e-1, 10.0)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
    gpc = GaussianProcessClassifier(kernel=kernel, random_state=42, )

    # Fit GP classifier
    gpc.fit(X_train, y_train)
    print(f"Fitted kernel: {gpc.kernel_}")

    # Predict on everything and compute metrics
    y_pred = gpc.predict(X_test)
    y_pred_train = gpc.predict(X_train)
    print(f"Accuracy = {np.sum(y_pred == y_test) / len(y_test)}")
    print(f"Precision = {np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_pred == 1)}")
    print(f"Recall = {np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1)}")

    plot_time_series_probs(X_test, X_train, gpc, test_run_names, test_runs, train_run_names, train_runs, y_test,
                           y_train)

    plot_decision_boundary(gpc, X_train, y_train, X_test, y_test)


    # TODO: how about we give the GPC some trajectory information?! Such as the current velocity in the latent space.


def plot_time_series_probs(X_test, X_train, gpc, test_run_names, test_runs, train_run_names, train_runs, y_test,
                           y_train):
    y_pred = gpc.predict_proba(X_test)
    y_pred_train = gpc.predict_proba(X_train)
    train_run_to_samples = defaultdict(list)
    train_run_to_y_pred = defaultdict(list)
    train_run_to_labels = defaultdict(list)
    for i in range(len(X_train)):
        train_run_to_labels[train_run_names[i]].append(y_train[i])
        train_run_to_samples[train_run_names[i]].append(X_train[i])
        train_run_to_y_pred[train_run_names[i]].append(y_pred_train[i][1])
    test_run_to_samples = defaultdict(list)
    test_run_to_y_pred = defaultdict(list)
    test_run_to_labels = defaultdict(list)
    for i in range(len(X_test)):
        test_run_to_labels[test_run_names[i]].append(y_test[i])
        test_run_to_samples[test_run_names[i]].append(X_test[i])
        test_run_to_y_pred[test_run_names[i]].append(y_pred[i][1])
    # Plot predicted probabilities as a function of index for each run
    plt.figure(figsize=(8, 4))
    for run in train_runs:
        ignition = train_run_to_labels[run][0]
        plt.plot(np.arange(len(train_run_to_samples[run])), train_run_to_y_pred[run], ls=":",
                 color="C1" if ignition else "C0")
        plt.xlabel("Time step")
        plt.ylabel("Predicted probability of ignition")
    for run in test_runs:
        ignition = test_run_to_labels[run][0]
        plt.plot(np.arange(len(test_run_to_samples[run])), test_run_to_y_pred[run], ls="-",
                 color="C1" if ignition else "C0")
    plt.plot([], [], ls=":", color="C0", label="Train (non-ignited)")
    plt.plot([], [], ls=":", color="C1", label="Train (ignited)")
    plt.plot([], [], ls="-", color="C0", label="Test (non-ignited)")
    plt.plot([], [], ls="-", color="C1", label="Test (ignited)")
    plt.legend()
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.savefig("output/gp_ip_timeseries.png")
    plt.show()


def plot_decision_boundary(gpc, X_train, y_train, X_test, y_test):
    # Set up grid
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Predict probs
    Z = gpc.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)

    # Predict variance


    # Plot
    plt.figure(figsize=(8, 8))

    im = plt.contourf(xx, yy, Z, cmap="jet", alpha=0.8)
    plt.colorbar(im, label="Probability of ignition")

    # Plot positive points with "o"
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='black', marker='o', label="Ignition training point")
    # Plot negative points with "x"
    plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='black', marker='x', label="Non-ignition training point")
    plt.legend()

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Gaussian Process Classifier decision boundary")
    plt.savefig("output/gp_ip_decision_boundary.png")
    plt.show()





if __name__ == '__main__':
    main()