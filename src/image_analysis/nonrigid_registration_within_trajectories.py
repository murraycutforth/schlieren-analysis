"""
In this script we register frames within a run to each other, aiming to see if the determinant of Jacobian of the
deformation field can be used to identify ignition events.

Author: Murray Cutforth
Date: 29/01/2024

"""

from itertools import pairwise
import SimpleITK as sitk

from skimage import filters
import numpy as np
import matplotlib.pyplot as plt

from src.utils.utils import load_cached_schlieren_images
from src.utils.data_processing import predict_ignition_using_final_frames
from src.utils.paths import output_dir


def main():
    run_to_image = load_cached_schlieren_images()
    run_to_ignition_status = {k: v for k, v in zip(run_to_image.keys(), predict_ignition_using_final_frames(run_to_image))}

    subregion = (slice(140, 180), slice(120, 160))

    all_det_jac_vals = []

    for run, images in run_to_image.items():

        registration_results = []
        for im1, im2 in pairwise(images):
            registration_results.append(register_image_pair(im2, im1))

        plot_registration_results(images, registration_results, run)
        plot_registration_results_zoomed(images, registration_results, run)

        avg_values = []
        for i, (im, transform) in enumerate(registration_results):
            det = get_deformation_jacobian_det(images[i], transform)
            avg_values.append(det[subregion].mean())

        all_det_jac_vals.append(avg_values)

    fig = plt.figure(figsize=(8, 8))
    for run, vals in zip(run_to_image.keys(), all_det_jac_vals):
        line_type = '-' if run_to_ignition_status[run] else '--'
        color = 'r' if run_to_ignition_status[run] else 'b'
        plt.plot(vals, linestyle=line_type, color=color)

    # Legend for ignitied / not ingited
    plt.plot([], label='Ignited', color="r", linestyle='-')
    plt.plot([], label='Not ignited', color="b", linestyle='--')
    plt.legend()

    plt.xlabel("Frame number")
    plt.ylabel("Average value of det(Jacobian) in kernel")
    plt.show()
    fig.savefig(output_dir() / f"all_det_jac_registration_results.png")
    plt.close(fig)


def plot_registration_results(images, registration_results, run, ncols=12):
    fig, axs = plt.subplots(4, ncols, sharey=True, sharex=True, figsize=(int(ncols * 1.5), 10))

    for i, ax in enumerate(axs[0, :]):
        ax.axis('off')
        ax.imshow(images[i].T, cmap='gray', origin='lower')
        ax.set_title(f"Frame {i}")

    axs[1, 0].axis('off')
    axs[1, 0].annotate("Registered images:", xy=(0.5, 0.5), xycoords='axes fraction', ha='center', va='center')
    for i, ax in enumerate(axs[1, 1:]):
        ax.axis('off')
        ax.imshow(registration_results[i][0].T, cmap='gray', origin='lower')
        ax.set_title(f"{i} to {i + 1}")

    axs[2, 0].axis('off')
    axs[2, 0].annotate("Registration error:", xy=(0.5, 0.5), xycoords='axes fraction', ha='center', va='center')
    for i, ax in enumerate(axs[2, 1:]):
        ax.axis('off')
        ax.imshow(np.abs(images[i+1].T - registration_results[i][0].T), cmap='jet', origin='lower', vmin=0, vmax=images[0].max() / 2)

    axs[3, 0].axis('off')
    axs[3, 0].annotate("Determinant of jacobian:", xy=(0.5, 0.5), xycoords='axes fraction', ha='center', va='center')
    for i, ax in enumerate(axs[3, 1:]):
        ax.axis('off')
        det = get_deformation_jacobian_det(images[i], registration_results[i][1])
        ax.imshow(det.T, cmap='jet', origin='lower', vmin=-5, vmax=5)

    fig.suptitle(f"Run {run}")
    fig.tight_layout()
    fig.savefig(output_dir() / f"run_{run}_registration_results.png")
    plt.close(fig)


def plot_det_jacb_vs_time(images, registration_results, run):
    subregion = (slice(140, 180), slice(120, 160))
    avg_values = []

    for i, (im, transform) in enumerate(registration_results):
        det = get_deformation_jacobian_det(images[i], transform)
        avg_values.append(det[subregion].mean())


def plot_registration_results_zoomed(images, registration_results, run):
    fig, axs = plt.subplots(4, 12, sharey=True, sharex=True, figsize=(14, 6))

    for i, ax in enumerate(axs[0, :]):
        ax.axis('off')
        ax.imshow(images[i][135:185, 115:165].T, cmap='gray', origin='lower')
        ax.set_title(f"Frame {i}")

    axs[1, 0].axis('off')
    axs[1, 0].annotate("Registered images:", xy=(0.5, 0.5), xycoords='axes fraction', ha='center', va='center')
    for i, ax in enumerate(axs[1, 1:]):
        ax.axis('off')
        ax.imshow(registration_results[i][0][135:185, 115:165].T, cmap='gray', origin='lower')
        ax.set_title(f"{i} to {i + 1}")

    axs[2, 0].axis('off')
    axs[2, 0].annotate("Registration error:", xy=(0.5, 0.5), xycoords='axes fraction', ha='center', va='center')
    for i, ax in enumerate(axs[2, 1:]):
        ax.axis('off')
        ax.imshow(np.abs(images[i+1][135:185, 115:165].T - registration_results[i][0][135:185, 115:165].T), cmap='jet', origin='lower', vmin=0, vmax=images[0].max() / 2)

    axs[3, 0].axis('off')
    axs[3, 0].annotate("Determinant of jacobian:", xy=(0.5, 0.5), xycoords='axes fraction', ha='center', va='center')
    for i, ax in enumerate(axs[3, 1:]):
        ax.axis('off')
        det = get_deformation_jacobian_det(images[i], registration_results[i][1])
        ax.imshow(det[135:185, 115:165].T, cmap='jet', origin='lower', vmin=-5, vmax=5)

    fig.suptitle(f"Run {run}")
    fig.tight_layout()
    fig.savefig(output_dir() / f"run_{run}_registration_results_zoomed.png")
    plt.close(fig)


def register_image_pair(fixed_im, moving_im) -> tuple[np.ndarray, sitk.Transform]:
    """Given two numpy arrays, register them using SimpleITK, and return the registered moving image"""
    fixed_image = sitk.GetImageFromArray(fixed_im)
    moving_image = sitk.GetImageFromArray(moving_im)

    registration_method = sitk.ImageRegistrationMethod()

    # Determine the number of BSpline control points using the physical
    # spacing we want for the finest resolution control grid.
    grid_physical_spacing = [5, 5]
    image_physical_size = [size*spacing for size,spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]
    mesh_size = [int(image_size/grid_spacing + 0.5) \
                 for image_size,grid_spacing in zip(image_physical_size,grid_physical_spacing)]

    # The starting mesh size will be 1/4 of the original, it will be refined by
    # the multi-resolution framework.
    mesh_size = [int(sz + 0.5) for sz in mesh_size]

    initial_transform = sitk.BSplineTransformInitializer(image1=fixed_image,
                                                         transformDomainMeshSize=mesh_size, order=3)
    # Instead of the standard SetInitialTransform we use the BSpline specific method which also
    # accepts the scaleFactors parameter to refine the BSpline mesh. In this case we start with
    # the given mesh_size at the highest pyramid level then we double it in the next lower level and
    # in the full resolution image we use a mesh that is four times the original size.
    registration_method.SetInitialTransformAsBSpline(initial_transform,
                                                     inPlace=False,
                                                     scaleFactors=[1])

    #registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    #registration_method.SetOptimizerAsLBFGS2(solutionAccuracy=1e-3, numberOfIterations=50, deltaConvergenceTolerance=1e-2)
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetOptimizerAsLBFGS2(numberOfIterations=100, deltaConvergenceTolerance=1e-2)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)

    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInterpolator(sitk.sitkLinear)

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
