<h1>Counting Of Spots in Marker Indicator Cells (COSMIC)</h1>

[![License](https://img.shields.io/pypi/l/napari-accelerated-pixel-and-object-classification.svg?color=green)](https://github.com/adiezsanchez/intestinal_organoid_yolov8/blob/main/LICENSE)
[![Development Status](https://img.shields.io/pypi/status/napari-accelerated-pixel-and-object-classification.svg)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Alpha)

This repository provides tools in the form of interactive Jupyter notebooks to count spots inside nuclei of cell marker positive and total cells. As an example we will be counting DNA damage foci inside microglia and astrocyte marker positive cells.

![workflow](./images/workflow.png)

<h2>Instructions</h2>

1. Create a raw_data directory inside the microglia_dna_damage folder to store all of your acquired images. In our case .lsm files acquired with a Zeiss microscope. This particular tools works with 3-channel images but is easy to adapt to multiple channels.

2. (Optional) Train your own Object and Pixel (semantic) APOC classifiers to detect spots and cell marker as shown in 0_train_dna_damage_segmenter.ipynb and 0_train_glia_semantic_classifier.ipynb. An example of how to do that using Napari-Assistant can be found [here](https://github.com/adiezsanchez/intestinal_organoid_brightfield_analysis/blob/main/1_train_and_setup.ipynb).

3. Open 1_image_analysis.ipynb and define the analysis parameters. Here's an explanation of what each parameter means and does during the analysis pipeline:

<h3>Nuclei segmentation</h3>

1. Gaussian smoothing blurs the input nuclei image so later on the Cellpose algorithm does not focus in bright spots inside the nuclei as separate objects. The amount of blurrying/smoothing can be controlled by the gaussian_sigma parameter (default = 1).

![nuclei_segmentation_gs6](./images/nuclei_seg_gs1.png)

The higher the gaussian_sigma values the increased chance of close sitting nuclei being detected as a single entity during Cellpose segmentation (see bright green nuclei below, gaussian_sigma = 6). On the other hand very low gaussian_sigma can result in incorrect segmentations or loss of nuclei entities. You will have to manually adjust this value according to your images.

![nuclei_segmentation_gs1](./images/nuclei_seg_gs6.png)

2. After gaussian smoothing a normalization step of contrast stretching is applied so Cellpose segmentation does not focus in dimmer vs more intense nuclei and misses detection of some.

3. During Cellpose 2.0 nuclei segmentation you can define the cellpose_nuclei_diameter values. This value corresponds to the diameter in pixels of the nuclei present in your image. Helps Cellpose adjust nuclei mask predictions.

4. After nuclei prediction and using [.cle](https://github.com/clEsperanto/pyclesperanto_prototype) functions, we dilate nuclei labels to make sure the spots we want to quantify are sitting inside or touching the nuclei mask. You can define the amount of dilation by modifying the dilation_radius_nuclei value.

5. Finally a nuclei label erosion of radius 1 is performed to avoid merging touching nuclei object upon binarization (needed to check which nuclei are cell marker positive - CM+).

<h3>Cell marker segmentation</h3>

<h2>Environment setup instructions</h2>

1. In order to run these Jupyter notebooks and .py scripts you will need to familiarize yourself with the use of Python virtual environments using Mamba. See instructions [here](https://biapol.github.io/blog/mara_lampert/getting_started_with_mambaforge_and_python/readme.html).

2. Then you will need to create a virtual environment using the command below or from the .yml file in the envs folder (recommended, see step 3):

   <code>mamba create -n microglia python=3.9 devbio-napari cellpose pytorch torchvision plotly pyqt -c conda-forge -c pytorch</code>

3. To recreate the venv from the environment.yml file stored in the envs folder (recommended) navigate into the envs folder using <code>cd</code> in your console and then execute:

   <code>mamba env create -f environment.yml</code>

4. Activate it by typing in the console:

   <code>mamba activate microglia</code>

5. Then launch Jupyter lab to interact with the code by typing:

   <code>jupyter lab</code>
