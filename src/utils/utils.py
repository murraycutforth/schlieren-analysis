import numpy as np
from sklearn.preprocessing import StandardScaler

from src.utils.paths import processed_data_dir


def load_cached_schlieren_images() -> dict[str, np.ndarray]:
    """Load the cached schlieren images
    """
    run_to_image = {}
    for npz_file in processed_data_dir().glob('*.npz'):
        run = npz_file.stem
        data = np.load(npz_file)
        run_to_image[run] = data["schlieren_image"]

    # Manually remove a bad frame from run 39 frame 17
    run_to_image["schlieren_COARSE_sim_39"] = np.delete(run_to_image["schlieren_COARSE_sim_39"], 17, axis=0)

    print(f"Loaded {len(run_to_image)} runs, with shapes {[v.shape for v in run_to_image.values()]}")
    print(f"Memory footprint of data = {sum([v.nbytes for v in run_to_image.values()]) / 1024 / 1024} MB")

    return run_to_image


def construct_full_data_matrix(run_to_image, whiten=True):
    X = np.stack([im.flatten() for v in run_to_image.values() for im in v], axis=0)
    if whiten:
        X = StandardScaler().fit_transform(X)
    return X
