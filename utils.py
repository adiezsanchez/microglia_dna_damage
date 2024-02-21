from pathlib import Path
from tqdm import tqdm
import napari
import tifffile
import numpy as np
from cellpose import models
import skimage
from skimage import measure, exposure
from scipy.ndimage import binary_erosion, binary_dilation, label
import pyclesperanto_prototype as cle
from apoc import ObjectSegmenter, PixelClassifier
import pandas as pd
import os


# Function to count particles in nuclei
def count_particles_in_nuclei(labeled_particles, labeled_nuclei, num_nuclei):
    particle_counts = np.zeros(num_nuclei, dtype=int)
    for i in range(1, num_nuclei + 1):
        nuclei_mask = labeled_nuclei == i
        # Count the number of unique particles in the nucleus
        unique_particles = np.unique(labeled_particles[nuclei_mask])
        # Exclude the background label (0)
        particle_counts[i - 1] = len(unique_particles[unique_particles > 0])
    return particle_counts


# Function to analyze images
def analyze_images(
    images,
    cellpose_nuclei_diameter,
    gaussian_sigma,
    dilation_radius_nuclei,
    glia_channel_threshold,
    dna_damage_segmenter_version,
    glia_segmenter,
    glia_segmenter_version,
    glia_nuclei_colocalization_erosion,
    dna_damage_erosion,
):
    model = models.Cellpose(gpu=True, model_type="nuclei")
    stats = []

    for image in tqdm(images):
        # Extract filename
        file_path = Path(image)
        filename = file_path.stem

        # Read the image file
        img = tifffile.imread(image)

        try:

            # Extract the stack containing the nuclei (0), dna_damage (1) and microglia channel (2)
            nuclei_stack = img[:, 0, :, :]
            dna_damage_stack = img[:, 1, :, :]
            microglia_stack = img[:, 2, :, :]

            # Perform maximum intensity projections
            nuclei_mip = np.max(nuclei_stack, axis=0)
            dna_damage_mip = np.max(dna_damage_stack, axis=0)
            microglia_mip = np.max(microglia_stack, axis=0)

            # Create a copy of nuclei_mip
            input_img = nuclei_mip.copy()

            # Might need to perform a Gaussian-blur before
            post_gaussian_img = skimage.filters.gaussian(
                input_img, sigma=gaussian_sigma
            )

            # Apply Contrast Stretching to improve Cellpose detection of overly bright nuclei
            p2, p98 = np.percentile(post_gaussian_img, (2, 98))
            img_rescale = exposure.rescale_intensity(
                post_gaussian_img, in_range=(p2, p98)
            )

            # Predict nuclei nuclei_masks using cellpose
            nuclei_masks, flows, styles, diams = model.eval(
                img_rescale,
                diameter=cellpose_nuclei_diameter,
                channels=[0, 0],
                net_avg=False,
            )

            # Dilate nuclei to make sure the dna_damage_foci objects sit inside
            dilated_nuclei_masks = cle.dilate_labels(
                nuclei_masks, radius=dilation_radius_nuclei
            )
            # Erode dilated_nuclei_masks to obtain separate objects upon binarization
            eroded_nuclei_masks = cle.erode_labels(dilated_nuclei_masks, radius=1)

            # Set a threshold value for the pixels in microglia channel
            threshold = glia_channel_threshold  # Based on the microglia marker intensity across images

            # Create a new array with the same shape as nuclei_masks, initialized with False
            result_array = np.full_like(nuclei_masks, False, dtype=bool)

            if glia_segmenter:
                # Predict glia mask using the semantic_segmenter instead of thresholding
                segmenter = PixelClassifier(
                    opencl_filename=f"./semantic_segmenters/microglia_segmenter_v{glia_segmenter_version}.cl"
                )
                above_threshold_indices = segmenter.predict(image=microglia_mip)
                # Extract ndarray from OCLArray
                above_threshold_indices = cle.pull(above_threshold_indices)
                # Transform into a boolean array
                above_threshold_indices = above_threshold_indices > 1
            else:
                # Find indices where values in values_array are above the threshold
                above_threshold_indices = microglia_mip > threshold

            # Update the corresponding positions in the result_array based on the mask_array
            nuclei_masks_bool = nuclei_masks.astype(
                bool
            )  # Make a boolean copy of nuclei_masks to be able to use logical operators
            result_array[nuclei_masks_bool & above_threshold_indices] = True

            # Convert the boolean array to a binary array
            binary_result_array = result_array.astype(int)

            # Now, result_array contains True where both conditions are satisfied, and False otherwise
            # viewer.add_labels(binary_result_array, name=f"{filename}_nuclei+glia_coloc")

            # Erode binary_result_array to get rid of small nuclei pixels colocalizing with glia branches
            # Set the structuring element for erosion
            structuring_element = np.ones(
                (
                    glia_nuclei_colocalization_erosion,
                    glia_nuclei_colocalization_erosion,
                ),
                dtype=bool,
            )  # You can adjust the size and shape

            # Perform binary erosion
            eroded_array = binary_erosion(
                binary_result_array, structure=structuring_element
            )

            # Now I want to recover just the nuclei_masks that are sitting on top of binary_results_array
            labels, num_labels = measure.label(nuclei_masks, return_num=True)

            # Create an array of indices corresponding to the True values in binary_result_array
            true_indices = np.where(eroded_array)

            # Initialize an array to store labels for each processed region
            processed_region_labels = np.zeros_like(nuclei_masks, dtype=int)

            # Label index for processed regions
            label_index = 1

            # Iterate over each connected component
            for label_ in range(1, num_labels + 1):
                # Extract the region corresponding to the current label
                region = labels == label_

                # Check if any True value in binary_result_array is present in the region
                if np.any(region[true_indices]):
                    # Assign a unique label to the processed region
                    processed_region_labels[region] = label_index
                    label_index += 1

            # viewer.add_labels(processed_region_labels, name=f"{filename}_glia+_nuclei_mask")

            # Dilate processed_regions to make sure the dna_damage_foci objects sit inside
            dilated_glia_pos_nuclei = cle.dilate_labels(
                processed_region_labels, radius=dilation_radius_nuclei
            )
            # Erode processed_regions to obtain separate objects upon binarization
            eroded_glia_pos_nuclei = cle.erode_labels(dilated_glia_pos_nuclei, radius=1)

            # Apply object segmenter from APOC
            segmenter = ObjectSegmenter(
                opencl_filename=f"./object_segmenters/dna_damage_object_segmenter_v{dna_damage_segmenter_version}.cl"
            )
            dna_damage_masks = segmenter.predict(image=dna_damage_mip)

            # Erode dna_damage_masks to get rid of small artifacts
            # Set the structuring element for erosion
            structuring_element_damage = np.ones(
                (dna_damage_erosion, dna_damage_erosion), dtype=bool
            )  # You can adjust the size and shape

            # Perform binary erosion
            eroded_dna_damage = binary_erosion(
                dna_damage_masks, structure=structuring_element_damage
            )

            # Perform binary dilation to fill holes
            dilated_dna_damage = binary_dilation(
                eroded_dna_damage, structure=structuring_element_damage
            )

            # Label the DNA damage particles
            labeled_dna_damage, num_dna_damage = label(dilated_dna_damage)

            # Label the nuclei in eroded_glia_pos_nuclei and eroded_nuclei_masks
            labeled_glia_nuclei, num_glia_nuclei = label(eroded_glia_pos_nuclei)
            labeled_nuclei, num_nuclei = label(eroded_nuclei_masks)

            # Count particles in glia nuclei
            particles_in_glia_nuclei = count_particles_in_nuclei(
                labeled_dna_damage, labeled_glia_nuclei, num_glia_nuclei
            )

            # Count particles in other nuclei
            particles_in_nuclei = count_particles_in_nuclei(
                labeled_dna_damage, labeled_nuclei, num_nuclei
            )

            # The arrays particles_in_glia_nuclei and particles_in_nuclei now contain the counts of DNA damage particles
            # in each nucleus of dilated_glia_pos_nuclei and dilated_nuclei_masks respectively

            # Create a dictionary containing all extracted info per image
            stats_dict = {
                "filename": filename,
                "avg_dna_damage_foci/glia_+": np.mean(particles_in_glia_nuclei),
                "avg_dna_damage_foci/glia_+_damage_+": np.mean(
                    particles_in_glia_nuclei[particles_in_glia_nuclei != 0]
                ),
                "avg_dna_damage_foci/all_nuclei": np.mean(particles_in_nuclei),
                "avg_dna_damage_foci/all_nuclei_damage_+": np.mean(
                    particles_in_nuclei[particles_in_nuclei != 0]
                ),
                "nr_+_dna_damage_glia_nuclei": np.count_nonzero(
                    particles_in_glia_nuclei
                ),
                "nr_-_dna_damage_glia_nuclei": (particles_in_glia_nuclei == 0).sum(),
                "nr_glia_+_nuclei": processed_region_labels.max(),
                "nr_total_nuclei": nuclei_masks.max(),
                "%_dna_damage_signal": (
                    np.sum(dilated_dna_damage) / dilated_dna_damage.size
                )
                * 100,
                "%_glia+_signal": (
                    np.sum(above_threshold_indices) / above_threshold_indices.size
                )
                * 100,
            }

            stats.append(stats_dict)

        except IndexError:
            print(f"Filename: {filename} displays an IndexError")
            pass

    df = pd.DataFrame(stats)

    # Define output folder for results
    results_folder = "./results/"

    # Create the necessary folder structure if it does not exist
    try:
        os.mkdir(str(results_folder))
        print(f"Output folder created: {results_folder}")
    except FileExistsError:
        print(f"Output folder already exists: {results_folder}")

    if glia_segmenter:
        df.to_csv(
            f"./results/results_cellpdia{cellpose_nuclei_diameter}_sigma{gaussian_sigma}_dilrad{dilation_radius_nuclei}_dnad_obj_seg_v{dna_damage_segmenter_version}_gliaero{glia_nuclei_colocalization_erosion}_glia_sem_seg_v{glia_segmenter_version}_dnadero{dna_damage_erosion}.csv",
            index=False,
        )
    else:
        df.to_csv(
            f"./results/results_cellpdia{cellpose_nuclei_diameter}_sigma{gaussian_sigma}_dilrad{dilation_radius_nuclei}_dnad_obj_seg_v{dna_damage_segmenter_version}_gliaero{glia_nuclei_colocalization_erosion}_gliathr{glia_channel_threshold}_dnadero{dna_damage_erosion}.csv",
            index=False,
        )

    return df
