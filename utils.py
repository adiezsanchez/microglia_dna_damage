from pathlib import Path
import re
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
import plotly.express as px


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
    dataset,
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

            # Count particles in all nuclei
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
                "nr_+_dna_damage_all_nuclei": np.count_nonzero(particles_in_nuclei),
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
                "damage_load_ratio_glia_+_cells": (
                    np.count_nonzero(particles_in_glia_nuclei) / processed_region_labels.max()
                ),
                "damage_load_ratio_all_cells": (
                    np.count_nonzero(particles_in_nuclei) / nuclei_masks.max()
                ),
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
            f"./results/{dataset}_results_cellpdia{cellpose_nuclei_diameter}_sigma{gaussian_sigma}_dilrad{dilation_radius_nuclei}_dnad_obj_seg_v{dna_damage_segmenter_version}_gliaero{glia_nuclei_colocalization_erosion}_glia_sem_seg_v{glia_segmenter_version}_dnadero{dna_damage_erosion}.csv",
            index=False,
        )
    else:
        df.to_csv(
            f"./results/{dataset}_results_cellpdia{cellpose_nuclei_diameter}_sigma{gaussian_sigma}_dilrad{dilation_radius_nuclei}_dnad_obj_seg_v{dna_damage_segmenter_version}_gliaero{glia_nuclei_colocalization_erosion}_gliathr{glia_channel_threshold}_dnadero{dna_damage_erosion}.csv",
            index=False,
        )

    return df


def read_results_csv(dataset, csv_path):

    if dataset == "microglia":
        mouse_id_csv_path = "./mouse_ids_Iba1.csv"
        print(f"The following dataset will be analyzed: {dataset}")

    elif dataset == "astrocyte":
        mouse_id_csv_path = "./mouse_ids_GFAP.csv"
        print(f"The following dataset will be analyzed: {dataset}")
    else:
        print("Please define an existing dataset")

    # Read the CSV file
    df = pd.read_csv(csv_path)
    df_mouse_id = pd.read_csv(mouse_id_csv_path, delimiter=";", encoding="UTF-8")

    # Convert the index into a column
    df.reset_index(inplace=True)

    # Extract 'tissue_location'
    df["tissue_location"] = df["filename"].str.split("_40X_").str[-1]

    # Extract 'staining_id'
    df["staining_id"] = df["filename"].str.extract("(\d+)_40X")

    # Process staining_ids into numeric to merge based on staining_id values
    df["staining_id"] = pd.to_numeric(df["staining_id"], errors="coerce")
    df_mouse_id["staining_id"] = pd.to_numeric(
        df_mouse_id["staining_id"], errors="coerce"
    )

    # Merge both processed_results_df and mouse_id dataframes on staining_id
    merged_df = pd.merge(df, df_mouse_id, on="staining_id")

    # Display the first few rows of the DataFrame
    merged_df.head()

    return df, df_mouse_id, merged_df


def extract_analysis_parameters(csv_path):
    "Takes the results.csv path as an input and extracts the analysis parameters using regular expressions"

    # Extract analysis parameters from the CSV path
    extracted_values = re.findall(r"\d+", csv_path)

    # Dynamically assign the extracted values to variables
    if len(extracted_values) >= 7:
        cellpose_nuclei_diameter = int(extracted_values[0])
        gaussian_sigma = int(extracted_values[1])
        dilation_radius_nuclei = int(extracted_values[2])
        dna_damage_segmenter_version = int(extracted_values[3])
        glia_nuclei_colocalization_erosion = int(extracted_values[4])
        dna_damage_erosion = int(extracted_values[6])
        if "_glia_sem_seg_v" in str(csv_path):
            glia_segmenter = True
        else:
            glia_segmenter = False

    if glia_segmenter:
        glia_segmenter_version = int(extracted_values[5])
        glia_channel_threshold = None
        # Dinamically adjust plot titles
        parameters_title = f"cellpdia{cellpose_nuclei_diameter}_sigma{gaussian_sigma}_dilrad{dilation_radius_nuclei}_dnad_obj_seg_v{dna_damage_segmenter_version}_gliaero{glia_nuclei_colocalization_erosion}_glia_sem_seg_v{glia_segmenter_version}_dnadero{dna_damage_erosion}"
    else:
        glia_channel_threshold = int(extracted_values[5])
        glia_segmenter_version = None
        # Dinamically adjust plot titles
        parameters_title = f"cellpdia{cellpose_nuclei_diameter}_sigma{gaussian_sigma}_dilrad{dilation_radius_nuclei}_dnad_obj_seg_v{dna_damage_segmenter_version}_gliaero{glia_nuclei_colocalization_erosion}_gliathr{glia_channel_threshold}_dnadero{dna_damage_erosion}"

    # Print the assigned analysis parameters
    print(f"Cellpose nuclei diameter: {cellpose_nuclei_diameter}")
    print(f"Gaussian sigma: {gaussian_sigma}")
    print(f"Dilation radius nuclei: {dilation_radius_nuclei}")
    print(f"Dna damage segmenter version: {dna_damage_segmenter_version}")
    print(f"Glia erosion: {glia_nuclei_colocalization_erosion}")
    print(f"Glia threshold: {glia_channel_threshold}")
    print(f"Glia semantic segmentation version: {glia_segmenter_version}")
    print(f"DNA damage foci erosion: {dna_damage_erosion}")

    return (
        cellpose_nuclei_diameter,
        gaussian_sigma,
        dilation_radius_nuclei,
        dna_damage_segmenter_version,
        glia_nuclei_colocalization_erosion,
        glia_channel_threshold,
        glia_segmenter,
        glia_segmenter_version,
        dna_damage_erosion,
        parameters_title,
    )
    
def show_exploratory_data(df, dataset, parameters_title):
    
    # Create the plot
    fig = px.scatter(df, x='tissue_location', y='nr_+_dna_damage_glia_nuclei',
                    hover_data=['staining_id','index','filename'], title=f"Number of DNA damage+ {dataset.capitalize()}+ Nuclei by Tissue Location - {parameters_title}")
    # Show the plot
    fig.show()
    
    # Create the plot
    fig = px.scatter(df, x='tissue_location', y='avg_dna_damage_foci/glia_+',
                    hover_data=['staining_id','index','filename'], title=f"Average DNA damage foci in {dataset.capitalize()} Nuclei by Tissue Location - {parameters_title}")
    # Show the plot
    fig.show()
    
    # Create the plot
    fig = px.scatter(df, x='tissue_location', y='avg_dna_damage_foci/all_nuclei',
                    hover_data=['staining_id','index','filename'], title=f"Average DNA damage foci in All Nuclei by Tissue Location - {parameters_title}")
    # Show the plot
    fig.show()
    
    # Create the plot
    fig = px.scatter(df, x='tissue_location', y='nr_glia_+_nuclei',
                    hover_data=['staining_id','index','filename'], title=f"Nr of {dataset.capitalize()}+ nuclei by Tissue Location - {parameters_title}")
    # Show the plot
    fig.show()
    
    # Create the plot
    fig = px.scatter(df, x='tissue_location', y='nr_total_nuclei',
                    hover_data=['staining_id','index','filename'], title=f'Nr of total nuclei by Tissue Location - {parameters_title}')
    # Show the plot
    fig.show()
    
    # Create the plot
    fig = px.scatter(df, x='staining_id', y='nr_glia_+_nuclei',
                    hover_data=['staining_id','index','filename'], title=f'Nr of {dataset.capitalize()}+ nuclei by Sample - {parameters_title}')
    # Show the plot
    fig.show()
    
    # Create the plot
    fig = px.scatter(df, x='staining_id', y='%_dna_damage_signal',
                    hover_data=['staining_id','index','filename'], title=f'Dna damage mask area (QC) - {parameters_title}')
    # Show the plot
    fig.show()
    
    # Create the plot
    fig = px.scatter(df, x='staining_id', y='%_glia+_signal',
                    hover_data=['staining_id','index','filename'], title=f'{dataset.capitalize()} mask area (QC) - {parameters_title}')
    # Show the plot
    fig.show()
    
def determine_stain_quality(value, mean_value):
    """Determines staining quality, anything above 3 times the mean value is considered an outlier"""
    if value < (mean_value + mean_value*3):
        return "optimal"
    else:
        return "suboptimal"
    
def qc_filter_dataset(
    merged_df,
    dataset, 
    cellpose_nuclei_diameter, 
    gaussian_sigma, 
    dilation_radius_nuclei, 
    dna_damage_segmenter_version, 
    glia_nuclei_colocalization_erosion, 
    glia_channel_threshold, 
    glia_segmenter, 
    glia_segmenter_version, 
    dna_damage_erosion
):
    
    # Calculate mean area of the image occupied by glia+ signal
    glia_mask_area_mean = merged_df['%_glia+_signal'].mean() 

    # Calculate mean area of the image occupied by dna_damage_+ signal
    dna_damage_mask_area_mean = merged_df['%_dna_damage_signal'].mean() 

    # Print extracted values
    print(f"Glia_mask_area_%_mean: {glia_mask_area_mean}, Dna_damage_mask_area_%_mean: {dna_damage_mask_area_mean}")
    
    # Check stain quality for glia and create another column storing optimal or suboptimal if qc_passed or not    
    merged_df['glia_stain_quality_auto'] = merged_df['%_glia+_signal'].apply(lambda x: determine_stain_quality(x, glia_mask_area_mean))

    # Check stain quality for dna_damage and create another column storing optimal or suboptimal if qc_passed or not 
    merged_df['dna_damage_stain_quality_auto'] = merged_df['%_dna_damage_signal'].apply(lambda x: determine_stain_quality(x, dna_damage_mask_area_mean))

    # Check for both stain qualities and store True qc_passed if both are optimal
    merged_df['staining_qc_passed'] = (merged_df['glia_stain_quality_auto'] == 'optimal') & (merged_df['dna_damage_stain_quality_auto'] == 'optimal')

    # Group the DataFrame by 'staining_id' and check if all 'staining_qc_passed' values are True, otherwise set them all to False
    merged_df['staining_qc_passed'] = merged_df.groupby('staining_id')['staining_qc_passed'].transform('all')

    # Now, if all 'staining_qc_passed' values for the same 'staining_id' were True, the column will remain True; otherwise, it will be False
    merged_df.head()
    
    # Save the resulting Dataframe into a .csv file
    if glia_segmenter:
        merged_df.to_csv(
        f"./results/qc_{dataset}_cellpdia{cellpose_nuclei_diameter}_sigma{gaussian_sigma}_dilrad{dilation_radius_nuclei}_dnad_obj_seg_v{dna_damage_segmenter_version}_gliaero{glia_nuclei_colocalization_erosion}_glia_sem_seg_v{glia_segmenter_version}_dnadero{dna_damage_erosion}.csv",
        index=False,
        )
    else:
        merged_df.to_csv(
            f"./results/qc_{dataset}_cellpdia{cellpose_nuclei_diameter}_sigma{gaussian_sigma}_dilrad{dilation_radius_nuclei}_dnad_obj_seg_v{dna_damage_segmenter_version}_gliaero{glia_nuclei_colocalization_erosion}_gliathr{glia_channel_threshold}_dnadero{dna_damage_erosion}.csv",
            index=False,
        )
    
    return merged_df

def plot_technical_replicates(auto_filtered_df, categories, dataset, parameters_title):
    
    # Create the boxplot with ordered categories
    fig = px.box(auto_filtered_df, x='sex_tissue', y='avg_dna_damage_foci/glia_+',
                color='genotype',  # Different genotypes will be shown in different colors
                category_orders={'sex_tissue': categories},  # Ensuring the specified order
                title=f'DNA Damage Foci nr in All {dataset.capitalize()} Nuclei by Tissue Location and Genotype (separated by sex) - Auto stain QC - {parameters_title}')

    # Show the plot
    fig.show()

    # Create the boxplot with ordered categories
    fig = px.box(auto_filtered_df, x='sex_tissue', y='avg_dna_damage_foci/glia_+_damage_+',
                color='genotype',  # Different genotypes will be shown in different colors
                category_orders={'sex_tissue': categories},  # Ensuring the specified order
                title=f'DNA Damage Foci nr in Damage+ {dataset.capitalize()} Nuclei by Tissue Location and Genotype (separated by sex) - Auto stain QC - {parameters_title}')

    # Show the plot
    fig.show()
    
    # Create the boxplot with ordered categories
    fig = px.box(auto_filtered_df, x='sex_tissue', y='avg_dna_damage_foci/all_nuclei',
                color='genotype',  # Different genotypes will be shown in different colors
                category_orders={'sex_tissue': categories},  # Ensuring the specified order
                title=f'DNA Damage Foci nr in All Nuclei by Tissue Location and Genotype (separated by sex) - Auto stain QC - {parameters_title}')

    # Show the plot
    fig.show()
    
    # Create the boxplot with ordered categories
    fig = px.box(auto_filtered_df, x='sex_tissue', y='avg_dna_damage_foci/all_nuclei_damage_+',
                color='genotype',  # Different genotypes will be shown in different colors
                category_orders={'sex_tissue': categories},  # Ensuring the specified order
                title=f'DNA Damage Foci nr in Damage+ Nuclei by Tissue Location and Genotype (separated by sex) - Auto stain QC - {parameters_title}')

    # Show the plot
    fig.show()
    
    # Create the boxplot with ordered categories
    fig = px.box(auto_filtered_df, x='sex_tissue', y='damage_load_ratio_glia_+_cells',
                color='genotype',  # Different genotypes will be shown in different colors
                category_orders={'sex_tissue': categories},  # Ensuring the specified order
                title=f'Ratio of damaged {dataset.capitalize()} nuclei by Tissue Location and Genotype (separated by sex) - Auto stain QC - {parameters_title}')

    # Show the plot
    fig.show()
    
    # Create the boxplot with ordered categories
    fig = px.box(auto_filtered_df, x='sex_tissue', y='damage_load_ratio_all_cells',
                color='genotype',  # Different genotypes will be shown in different colors
                category_orders={'sex_tissue': categories},  # Ensuring the specified order
                title=f'Ratio of damaged nuclei by Tissue Location and Genotype (separated by sex) - Auto stain QC - {parameters_title}')

    # Show the plot
    fig.show()