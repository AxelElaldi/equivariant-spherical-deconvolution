# Equivariant Spherical Deconvolution
PyTorch implementation of the paper [Equivariant Spherical Deconvolution: Learning Sparse Orientation Distribution Functions from Spherical Data](https://arxiv.org/abs/2102.09462) from Axel Elaldi*, Neel Dey*, Heejong Kim and Guido Gerig (*equal contribution). Main application of this work is for diffusion MRI and fODF estimation, and can be extended to other deconvolution problem.

![alt text](https://github.com/AxelElaldi/equivariant-spherical-deconvolution/blob/esd/img/summary.png)


We use the spherical graph convolution from [DeepSphere](https://github.com/deepsphere/deepsphere-pytorch).

The response functions are given to the network and are stored as [spherical harmonic coefficients](https://en.wikipedia.org/wiki/Spherical_harmonics) (SHC). Since these signals are polar signals, every SHCs are nulls but the ones of order 0. Thus, a reponse function is a matrix of size **SxL**, where **S** is the number of input shells and **L** is the maximum spherical harmonic degree of the response functions. A response function file as a txt file with **S** rows and **L** columns ([MRtrix](https://mrtrix.readthedocs.io/en/3.0.1/concepts/spherical_harmonics.html) convention).

## 1. Getting started

Install the following library:

```
    conda create -n esd python=3.8
    source activate esd
    pip install git+https://github.com/epfl-lts2/pygsp.git@39a0665f637191152605911cf209fc16a36e5ae9#egg=PyGSP
    pip install numpy scipy matplotlib ipython jupyter pandas sympy nose
    pip install nibabel
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch # Tested for PyTorch 1.10
    pip install healpy
    pip install tensorboard

```

## 2. Prepare the data

```python:
    data_path = 'data_root'
```

In a root folder:
* Copy your diffusion MRI data (resp. the mask) as a nifti file under the name **features.nii** (**mask.nii**). 
* Copy your bvecs and bvals files under the names **bvecs.bvecs** and **bvals.bvals**.
* In the root folder, create a folder for the response functions, called **response_functions**. There, create a folder for each response function estimation algorithm you want to use. We will use as example folder the name **rf_algo**. In each algorithm folder, copy the white matter, grey matter, and CSF reponse function files under the names **wm_response.txt**, **gm_response.txt**, and **csf_response.txt**. We refer to [Mrtrix3](https://mrtrix.readthedocs.io/en/0.3.16/concepts/response_function_estimation.html) for different response function algorithms.

## 3. Short example
```python:
    from utils.sampling import HealpixSampling, ShellSampling
    from utils.dataset import Dataset
    from utils.response import load_response_function
    from model.model import Model
    from model.shutils import ComputeSignal
    from model.reconstruction import IsoSHConv
    import torch

    data_path = 'data_root' # Root path of the data
    rf_name = 'rf_algo' # Name of the response function estimation algorithm
    sh_degree = 20 # Max spherical harmonic degree of the estimated fODF
    n_side = 16 # Resolution of the healpix grid (must be a power of 2)
    depth = 5 # Depth of the U-Net
    wm, gm, csf = True, True, True # Use wm, gm and csf fODF to reconstruct the signal
    filter_start = 8 # Number of output features for the first convolution layer
    kernel_size = 3 # Kernel size in the U-Net
    normalize = True # Normalize the estimated fODF

    # Load the shell and the graph samplings
    shellSampling = ShellSampling(f'{data_path}/bvecs.bvecs', f'{data_path}/bvals.bvals', sh_degree=sh_degree, max_sh_degree=8) # V' vertices, S shells
    graphSampling = HealpixSampling(n_side, depth, sh_degree=sh_degree) # V vertices

    # Load the Polar filter used for the deconvolution
    polar_filter_equi, polar_filter_inva = load_response_function(f'{data_path}/response_functions/{rf_name}', wm=wm, gm=gm, csf=csf, max_degree=sh_degree, n_shell=len(shellSampling.shell_values)) # 1 x S x C (because 1 equivariant filter, wm), 2 x S x 1 (because 2 invariant filters, gm and csf)

    # Create the deconvolution model
    model = Model(polar_filter_equi, polar_filter_inva, shellSampling, graphSampling, filter_start, kernel_size, normalize)

    # Generate random signal
    batch_size = 16
    shell_vertices = len(shellSampling.vectors)
    input_feature = 1
    x = torch.rand(batch_size, input_feature, shell_vertices) # B x F x V'
    x_reconstructed, x_deconvolved_equi_shc, x_deconvolved_inva_shc = model(x) # B x V', B x 1 x C (because 1 equivariant output, wm), B x 2 x 1 (because 2 invariant output, gm and csf)

    # Compute a signal on the graphSampling grid from the spherical harmonic coefficients
    denseGrid_interpolate = ComputeSignal(torch.Tensor(graphSampling.sampling.SH2S))
    x_deconvolved_equi = denseGrid_interpolate(x_deconvolved_equi_shc) # B x 1 x V

    # Spherical harmonic convolution with a polar filter
    conv_equi = IsoSHConv(polar_filter_equi)
    x_convolved_equi_shc = conv_equi(x_deconvolved_equi_shc) # B x 1 x S x C
```


## 4. Shell sampling &harr; Graph sampling

In our work, we consider the shells as different spherical feature maps on the unit sphere. The network expects each feature maps to be sampled on the same set of vertices, which is usually not the case. To overcome this issue, the first module of the network is an interpolation from the shell sampling to the graph sampling. We used a spherical harmonic interpolation, but you can define your own interpolation following the example in [Interpolation](https://github.com/AxelElaldi/equivariant-spherical-deconvolution/blob/esd/model/interpolation.py).

### 4.1 Scheme sampling

The ShellSampling class computes, for each shell, the matrices to compute:
- The SHC from the shell signal (with max degree=min(max_sh_order, sh_degree, H), see Class implementation for the definition of H)
- The shell signal from the SHC (with max degree=sh_degree)

```python:
    sh_degree = 20
    shellSampling = ShellSampling(f'{data_path}/bvecs.bvecs', f'{data_path}/bvals.bvals', sh_degree=sh_degree, max_sh_degree=8)
```

### 4.2 Graph sampling

The GraphSampling class defines the graph structure used by the graph convolution. We use the healpix sampling because of its hierarchical structure that makes easier the pooling and unpooling operations. You can create your own GrahSampling class following the example in [GraphSampling](https://github.com/AxelElaldi/equivariant-spherical-deconvolution/blob/esd/utils/sampling.py) and your own pooling class following the example in [Pooling](https://github.com/AxelElaldi/equivariant-spherical-deconvolution/blob/esd/utils/pooling.py).
```python:
    n_side = 16 # The input and output signal are evaluated on an healpix grid of resolution 16
    depth = 5 # We use 5 spherical grid resolution in the network (equivalent to 4 spherical pooling and unpooling)
    sh_degree = 20
    shellSampling = HealpixSampling(n_side, depth, sh_degree=sh_degree)
```

## 5. Polar filters
We work with two different polar filters:
- Rotation invariant filter, i.e. a filter that can be described using only one spherical harmonic degree (for example the grey matter or the CSF response function are constant filters).
- Rotation equivariant filter, i.e. a filter that use more than one spherical harmonic degree to be described (for example the white matter response function).

We separate these two classes of filter to speed up the convolution between the filters and the deconvolved signal.
```python:
    rf_name = 'rf_algo'
    wm, gm, csf = True, True, True
    polar_filter_equi, polar_filter_inva = load_response_function(f'{data_path}/response_functions/{rf_name}', wm=wm, gm=gm, csf=csf, max_degree=sh_degree, n_shell=len(shellSampling.shell_values))
```

## 6. Deconvolution and Reconstruction model
We are now ready to create the deconvolution and reconstruction model, defined in [Model](https://github.com/AxelElaldi/equivariant-spherical-deconvolution/blob/esd/model/model.py).
```python:
    filter_start = 8
    kernel_size = 5
    normalize = True
    model = Model(polar_filter_equi, polar_filter_inva, shellSampling, graphSampling, filter_start, kernel_size, normalize)
```

### 6.1 Deconvolution module
The first part of the model is a [deconvolution](https://github.com/AxelElaldi/equivariant-spherical-deconvolution/blob/esd/model/deconvolution.py) module, itself decomposed in 5 parts:
- Interpolation of the raw input (living on the Shell Sampling) onto the graph samping. We use a spherical harmonic interpolation, but you can implement your own module [Interpolation](https://github.com/AxelElaldi/equivariant-spherical-deconvolution/blob/esd/model/interpolation.py).
- Deconvolution using a spherical graph U-Net. The model takes as input a voxel and output one spherical feature maps per polar filter. We use a Chebyshev convolution [Convolution](https://github.com/AxelElaldi/equivariant-spherical-deconvolution/blob/esd/model/graphconv.py).
- Separate the equivariant and invariant spherical feature maps, and reduce the invariant spherical maps to one scalar per maps (we use the sum operation).
- Compute the spherical harmonic coefficient of the equivariant and invariant outputs.
- (Optional) Normalize the spherical harmonic coefficients.

### 6.2 Reconstruction module
The second part of the model is a [reconstruction](https://github.com/AxelElaldi/equivariant-spherical-deconvolution/blob/esd/model/reconstruction.py) module using a spherical harmonic convolution, itself decomposed in 2 parts:
- Spherical convolution between the polar filters and the spherical maps.
- Reconstruction on the Shell Sampling.

(TO DO: implement the spatial spherical convolution)

## 7. Apply model
To start using a model, you can load your nifti image and mask:
```python:
    dataset = Dataset(f'{data_path}/features.nii', f'{data_path}/mask.nii')

    data = dataset.__getitem__(0)
    input = data['input']
    x_reconstructed, x_deconvolved_equi_shc, x_deconvolved_inva_shc = model(input)
```
The first output of the model is the reconstructed signal on the Shell Sampling. The second and third output are the spherical harmonic coefficients of the equivariant and invariant spherical maps.


## 8. Train a model
You can train a new model on your data using the following bash command:

```
    python train.py --data_path /path/to/data/  --batch_size 32 --lr 0.0017 --epoch 50  --filter_start 8 --sh_degree 20  --save_every 1 --loss_intensity L2 --loss_sparsity cauchy --loss_non_negativity L2 --sigma_sparsity 1e-05 --sparsity_weight 1e-4 --intensity_weight 1 --nn_fodf_weight 1 --wm --gm --csf --rf_name rf_algo --depth 5
```

## 9. Test a model
You can test a trained model on your data using the following bash command:

```
    python test.py --data_path /path/to/data/ --batch_size 32 --model_name model_name --epoch 10
```

## 10. Result
![alt text](https://github.com/AxelElaldi/equivariant-spherical-deconvolution/blob/esd/img/drawing-fodf_synth.png)
Qualitative synthetic (Sec. 3.1) results showing fODF estimation on 128-gradient 2-fiber samples with red arrows representing ground truth fibers and the heatmap showing model prediction. Row 1: CSD [1], Row 2: ESD (ours).

![alt text](https://github.com/AxelElaldi/equivariant-spherical-deconvolution/blob/esd/img/drawing-tract2.png)
Post-deconvolution tractography for [Tractometer](http://www.tractometer.org/ismrm_2015_challenge/) (single-shell). ESD demonstrates clearer streamlines with lower noise as opposed to CSD. Readers are encouraged to zoom-in for visual inspection.

[1] Jeurissen, B., Tournier, J.D., Dhollander, T., Connelly, A., Sijbers, J.: Multi-tissue constrained spherical deconvolution for improved analysis of multi-shell diffusion mri data. NeuroImage 103, 411–426 (2014)

## Licence

Part of the graph convolution code comes from [DeepSphere](https://github.com/deepsphere/deepsphere-pytorch).

Please consider citing our paper if you find this repository useful.
```
    @inproceedings{elaldi2021equivariant,
    title={Equivariant Spherical Deconvolution: Learning Sparse Orientation Distribution Functions from Spherical Data},
    author={Elaldi, Axel and Dey, Neel and Kim, Heejong and Gerig, Guido},
    booktitle={International Conference on Information Processing in Medical Imaging},
    pages={267--278},
    year={2021},
    organization={Springer}
    }
```