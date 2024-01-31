from collections import defaultdict
from pathlib import Path
import SimpleITK as sitk

from skimage import filters
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils.paths import output_dir
from src.utils.utils import load_cached_schlieren_images
from src.utils.data_processing import predict_ignition_using_final_frames
from src.image_analysis.nonrigid_registration_within_trajectories import plot_registration_results


def main():
    run_to_image = load_cached_schlieren_images()
    run_names = [k for k in run_to_image.keys()]
    ignition_status = predict_ignition_using_final_frames(run_to_image)
    run_to_ignition_status = {k: v for k, v in zip(run_names, ignition_status)}

    run_to_image_ignition = {k: v for k, v in run_to_image.items() if run_to_ignition_status[k]}
    run_to_image_nonignition = {k: v for k, v in run_to_image.items() if not run_to_ignition_status[k]}
    assert len(run_to_image_ignition) == 7


    #plot_explained_variance(pca)
    #plot_principal_components(im_shape, pca)

    # Nonrigid registration of final frames together of ignition runs
    final_frames = [v[-1] for v in run_to_image_ignition.values()]
    reference_im = final_frames[6]

    # Register all images to the first one
    registration_results = []
    for im in final_frames:
        registration_results.append(register_image_pair(reference_im, im))

    plot_registration_results(final_frames, registration_results, "Ignition final frames", ncols=7)

    # Compute PCA of originals
    X = np.stack([v.flatten() for v in final_frames], axis=0)
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)

    # Compute PCA of registered images
    X_r = np.stack([v[0].flatten() for v in registration_results], axis=0)
    X_r = StandardScaler().fit_transform(X_r)
    pca_r = PCA(n_components=2)
    X_r = pca_r.fit_transform(X_r)

    # Now plot the PCA of the original and registered images
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].scatter(X[:, 0], X[:, 1])
    axs[0].set_title("PCA of original final frames")
    axs[0].set_xlabel("PC1")
    axs[0].set_ylabel("PC2")
    axs[1].scatter(X_r[:, 0], X_r[:, 1])
    axs[1].set_title("PCA of registered final frames")
    axs[1].set_xlabel("PC1")
    axs[1].set_ylabel("PC2")
    fig.savefig(output_dir() / "pca_original_vs_registered.png")
    plt.show()



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
    registration_method.SetOptimizerAsLBFGS2(numberOfIterations=200, deltaConvergenceTolerance=0.01)

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
