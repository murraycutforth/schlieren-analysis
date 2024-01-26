from collections import defaultdict
from pathlib import Path
import SimpleITK as sitk

from skimage import filters
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import load_cached_schlieren_images

PREPROCESSED_IMAGE_DIR = Path('../../data/preprocessed')


def main():
    run_to_image = load_cached_schlieren_images()
    run_names = [k for k in run_to_image.keys()]

    print(f"Loaded {len(run_to_image)} runs, with shapes {[v.shape for v in run_to_image.values()]}")
    print(f"Memory footprint of data = {sum([v.nbytes for v in run_to_image.values()]) / 1024 / 1024} MB")

    # plot_PCA_final_frame(run_to_image)

    # Next up: create ignition / non-ignition labels based on PCA and show images for first few PCs

    im_shape = next(iter(run_to_image.values())).shape[1:]
    pca = PCA(n_components=10)
    X = np.stack([v[-1].flatten() for v in run_to_image.values()], axis=0)
    X = StandardScaler().fit_transform(X)
    pca.fit(X)

    pca_2 = PCA(n_components=2)
    X_2 = pca_2.fit_transform(X)

    # Plot X_2
    plt.figure()
    plt.scatter(X_2[:, 0], X_2[:, 1])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of final frames")
    plt.show()

    # Classification of ignition / non-ignition using PCA1 threshold
    pca_1_threshold = 50
    ignition_indices = np.where(X_2[:, 0] > pca_1_threshold)[0]
    iginition_runs = [run_names[i] for i in ignition_indices]
    run_to_image_ignition = {k: v for k, v in run_to_image.items() if k in iginition_runs}
    run_to_image_nonignition = {k: v for k, v in run_to_image.items() if k not in iginition_runs}
    assert len(run_to_image_ignition) == 7


    #plot_explained_variance(pca)
    #plot_principal_components(im_shape, pca)

    # Nonrigid registration of final frames together of ignition runs
    final_frames = [v[-1] for v in run_to_image_ignition.values()]
    reference_im = final_frames[6]

    # Register all images to the first one
    registered_final_frames = []
    transforms = []
    for im in final_frames:
        print("Registering image")
        registered_image, transform = register_image_pair(reference_im, im)
        registered_final_frames.append(registered_image)
        transforms.append(transform)

    # Plot the registered images - first row is the original, second row is the registered, third row is the displacement field
    fig, axs = plt.subplots(3, 7, sharey=True, sharex=True)
    for i, ax in enumerate(axs[0, :].flat):
        ax.imshow(final_frames[i])
        ax.set_title(f"{[k for k in run_to_image_ignition][i]}")
    for i, ax in enumerate(axs[1, :].flat):
        ax.imshow(registered_final_frames[i])
    for i, ax in enumerate(axs[2, :].flat):
        det = get_deformation_jacobian_det(final_frames[i], transforms[i])
        ax.imshow(det, cmap='jet')

    axs[0, 0].set_ylabel(f"Original images")
    axs[1, 0].set_ylabel(f"Registered images")
    axs[2, 0].set_ylabel(f"Determinant of Jacobian of displacement field")
    fig.tight_layout()
    plt.show()

    # Now look at the PCA of the registered images compared to the originals
    X_registered = np.stack([v.flatten() for v in registered_final_frames], axis=0)
    X_registered = StandardScaler().fit_transform(X_registered)
    X_registered_2 = pca_2.transform(X_registered)

    # Plot X_2
    plt.figure()
    plt.scatter(X_2[:, 0], X_2[:, 1], label="Original")
    plt.scatter(X_registered_2[:, 0], X_registered_2[:, 1], label="Registered")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of final frames")
    plt.legend()
    plt.show()

    # Try combining the registered images with the non-ignition images, and add a gaussian blur filtering
    final_frames_nonignition = [filters.gaussian(v[-1], sigma=10) for v in run_to_image_nonignition.values()]
    registered_frames_ignition = [filters.gaussian(v, sigma=10) for v in registered_final_frames]
    combined_final_frames = final_frames_nonignition + registered_frames_ignition
    X_combined = np.stack([v.flatten() for v in combined_final_frames], axis=0)
    X_combined = StandardScaler().fit_transform(X_combined)
    pca_combined = PCA(n_components=2)
    X_combined_2 = pca_combined.fit_transform(X_combined)

    # Plot X_2
    plt.figure()
    plt.scatter(X_combined_2[:, 0], X_combined_2[:, 1])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of final frames")
    plt.show()



    




    # Option: try out nonrigid registration of final frames before PCA
    # Set up data for DMD


def plot_principal_components(im_shape, pca):
    # Plot the first 10 principal components
    fig, axs = plt.subplots(2, 5)
    for i, ax in enumerate(axs.flat):
        ax.imshow(pca.components_[i].reshape(im_shape).T)
        ax.set_title(f"PC{i}")
        ax.axis('off')
    plt.show()


def plot_explained_variance(pca):
    # Plot the cumulative explained variance ratio
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.ylim(0, 1)
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("Cumulative explained variance ratio")
    plt.show()


def register_image_pair(fixed_im, moving_im):
    """Given two numpy arrays, register them using SimpleITK, and return the registered moving image"""
    fixed_image = sitk.GetImageFromArray(fixed_im)
    moving_image = sitk.GetImageFromArray(moving_im)

    registration_method = sitk.ImageRegistrationMethod()

    # Determine the number of BSpline control points using the physical
    # spacing we want for the finest resolution control grid.
    grid_physical_spacing = [20, 20]
    image_physical_size = [size*spacing for size,spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]
    mesh_size = [int(image_size/grid_spacing + 0.5) \
                 for image_size,grid_spacing in zip(image_physical_size,grid_physical_spacing)]

    # The starting mesh size will be 1/4 of the original, it will be refined by
    # the multi-resolution framework.
    mesh_size = [int(sz/4 + 0.5) for sz in mesh_size]

    initial_transform = sitk.BSplineTransformInitializer(image1 = fixed_image,
                                                         transformDomainMeshSize = mesh_size, order=3)
    # Instead of the standard SetInitialTransform we use the BSpline specific method which also
    # accepts the scaleFactors parameter to refine the BSpline mesh. In this case we start with
    # the given mesh_size at the highest pyramid level then we double it in the next lower level and
    # in the full resolution image we use a mesh that is four times the original size.
    registration_method.SetInitialTransformAsBSpline(initial_transform,
                                                     inPlace=False,
                                                     scaleFactors=[1,2,4])

    #registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)

    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsLBFGS2(solutionAccuracy=1e-3, numberOfIterations=200, deltaConvergenceTolerance=0.01)

    final_transformation = registration_method.Execute(fixed_image, moving_image)

    # Get the registered image
    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transformation, sitk.sitkLinear, 0.0,
                                     moving_image.GetPixelID())
    moving_registered = sitk.GetArrayFromImage(moving_resampled)
    return moving_registered, final_transformation


def get_deformation_jacobian_det(image, transform):
    """Get the determinant of the Jacobian of the deformation field for the given image and transform"""
    image = sitk.GetImageFromArray(image)
    displ = sitk.TransformToDisplacementField(transform,
                                              sitk.sitkVectorFloat64,
                                              image.GetSize(),
                                              image.GetOrigin(),
                                              image.GetSpacing(),
                                              image.GetDirection())
    det = sitk.GetArrayFromImage(sitk.DisplacementFieldJacobianDeterminant(displ))
    return det


if __name__ == '__main__':
    main()
