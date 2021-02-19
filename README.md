# Equivariant Spherical Deconvolution
PyTorch implementation of the paper [Equivariant Spherical Deconvolution: Learning Sparse Orientation Distribution Functions from Spherical Data](https://arxiv.org/abs/2102.09462) from Axel Elaldi*, Neel Dey*, Heejong Kim and Guido Gerig (*equal contribution). Main application of this work is for diffusion MRI and fODF estimation.

## 1. Getting started

1. We use [DeepSphere](https://github.com/deepsphere/deepsphere-pytorch) for our convolution layers. Follow their installation instructions.

2. We use [Dipy](https://dipy.org/), [Healpy](https://healpy.readthedocs.io/en/latest/), [Scipy](https://www.scipy.org/), [Nibabel](https://nipy.org/nibabel/), [Matplotlib](https://matplotlib.org/stable/index.html), [Pandas](https://pandas.pydata.org/) and [Seaborn](https://seaborn.pydata.org/)

To run the CSD algorithms, you'll need to install [Mrtrix3](https://mrtrix.readthedocs.io/en/latest/installation/package_install.html) (SSST and MSMT) and [MRtrix3Tissue](https://3tissue.github.io/doc/ss3t-csd.html) (SSMT).

## 2. How to use ESD.
We provide notebooks to run the different components of the project.

### 2.1 Data preparation
Two options. Either simulate a **synthetic dataset** or work with a **real dataset**.

#### Synthetic dataset (notebook/2. Synthetic dataset creation):
We provide a script to create a Multi-Tissue synthetic dataset, which needs pre-computed Reponse Functions for each tissue and a gradients scheme. We provide examples of such gradient schemes and Response Functions in `scheme_example/`.

* Create the ground-truth data:

```
python generate_data/2_ground_truth_generation.py --max_tensor 3 --n_data 1000 --n_save 1000 --start 1 --path /mnt/archive/data/synthetic_data --rand_tensor --rand_angles --rand_vf_tissue
```

* Generate the synthetic dataset:

To generate the synthetic dataset, you will need a scheme (bvecs and bvals files) and the three tissue response functions matching the scheme bvals.

To use your own scheme, add in the folder `scheme_example/` a new directory, name it with the name of your scheme, add the `bvals.bvals` and the `bvecs.bvecs` files to it, and add the three response function files. The response functions files are the same as the ones from the `dwi2rf` Mrtrix function.

```
python generate_data/3_signal_generation.py --snr 20 --n_save 1000 --base_path /mnt/archive/data/synthetic_data --name_scheme 64_points_3_shell --rf_path scheme_example/64_points_3_shell
```

* Convert the pkl file into a nii file:

```
python generate_data/4_pkl_to_nii.py --path /mnt/archive/data/synthetic_data/64_points_3_shell/20_snr/gradients --dim
```

#### Real dataset (notebook/3. Real dataset preparation):

If you have a nifti file with diffusion MRI data, you can transform it such that ESD can learn from it. The best is to follow the notebook instructions.

### 2.2 Response function estimation (notebook/4. RF estimation)
Once you have a dataset, the first step is to estimate the tissue reponse functions. For more detail, you can read [Mrtrix3 documentation](https://mrtrix.readthedocs.io/en/latest/constrained_spherical_deconvolution/response_function_estimation.html). 

* Transform the nifti file into a mif file (mrtrix file)

```
python generate_data/5_nii_to_mif.py --path /mnt/archive/data/synthetic_data/64_points_3_shell/20_snr/gradients/data_cat --path_bvals_bvecs /mnt/archive/data/synthetic_data/64_points_3_shell/scheme
```

* Compute Tournier RF:

```
python generate_data/6_rf_estimation.py --path /mnt/archive/data/synthetic_data/64_points_3_shell/20_snr/gradients --method_name tournier --lmax 8
```

* Compute Dhollander RF

```
python generate_data/6_rf_estimation.py --path /mnt/archive/data/synthetic_data/64_points_3_shell/20_snr/gradients --method_name dhollander --lmax 10 --no_erode
```

if you use a real dataset, you should delete the flag --no_erode and add the flag --mask.

### 2.3 Compute the CSD fODF from mrtrix (notebook/5. Compute CSD)

You can run the CSD comparison experiments. It is needed if you want to use the data augmentation ESD training. Example for MSMT:

```
python compare_csd/1_mrtrix_csd.py --path /mnt/archive/data/synthetic_data/64_points_3_shell/20_snr/gradients --path_rf /mnt/archive/data/synthetic_data/64_points_3_shell/20_snr/gradients/response_functions/dhollander_10_None --model msmt --gm --csf
```

```
python generate_data/7_mif_to_nii.py --path /mnt/archive/data/synthetic_data/64_points_3_shell/20_snr/gradients/result/msmt_dhollander_10_None_None/fodf/fodf_cat/fodf.mif
```

### 2.4 Train ESD (notebook/6. Train model)

You can train the Multi-Shell Multi-Tissue ESD model:

```
python model/train.py --root_path /mnt/archive/data/synthetic_data --scheme 64_points_3_shell --snr 20 --batch_size 32 --lr 0.01 --epoch 50 --activation relu --normalization batch --pooling max --filter_start 8 --max_order 20 --interpolation sh --save_evry 1 --loss_intensity L2 --loss_sparsity cauchy --loss_non_negativity L2 --sigma_sparsity 1e-05 --sparsity_weight 0.001 --intensity_weight 1 --nn_fodf_weight 1 --wm_path /mnt/archive/data/synthetic_data/64_points_3_shell/20_snr/gradients/response_functions/dhollander_10_None/wm_response.txt --gm_path /mnt/archive/data/synthetic_data/64_points_3_shell/20_snr/gradients/response_functions/dhollander_10_None/gm_response.txt --csf_path /mnt/archive/data/synthetic_data/64_points_3_shell/20_snr/gradients/response_functions/dhollander_10_None/csf_response.txt --val --threshold 0.5 --sep_min 15 --max_fiber 3 --split_nb 1
```

### 2.5 Test ESD (notebook/7. Test model)

See the notebook for examples

### 2.6 Model performance (notebook/8. Model performance)
See the notebook for examples
