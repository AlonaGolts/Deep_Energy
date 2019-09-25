# Deep-Energy: Unsupervised Training of Deep Neural Networks

This is a TensorFlow implementation of our two papers:

[1] [Unsupervised Single Image Dehazing using Dark Channel Prior Loss](https://arxiv.org/abs/1812.07051)

[**Alona Golts**](https://il.linkedin.com/in/alona-golts-812b83b5), [**Daniel Freedman**](https://ai.google/research/people/DanielFreedman/), [**Michael Elad**](https://elad.cs.technion.ac.il/)

Abstract: Single image dehazing is a critical stage in many modern-day autonomous vision applications. Early prior-based methods often involved a time-consuming minimization of a hand-crafted energy function. Recent learning-based approaches utilize the representational power of deep neural networks (DNNs) to learn the underlying transformation between hazy and clear images. Due to inherent limitations in collecting matching clear and hazy images, these methods resort to training on synthetic data; constructed from indoor images and corresponding depth information. This may result in a possible domain shift when treating outdoor scenes. We propose a completely unsupervised method of training via minimization of the well-known, Dark Channel Prior (DCP) energy function. Instead of feeding the network with synthetic data, we solely use real-world outdoor images and tune the network's parameters by directly minimizing the DCP. Although our "Deep DCP" technique can be regarded as a fast approximator of DCP, it actually improves its results significantly. This suggests an additional regularization obtained via the network and learning process. Experiments show that our method performs on par with large-scale
supervised methods.

[2] [Deep-Energy: Unsupervised Training of Deep Neural Networks](https://arxiv.org/abs/1805.12355)

[**Alona Golts**](https://il.linkedin.com/in/alona-golts-812b83b5), [**Daniel Freedman**](https://ai.google/research/people/DanielFreedman/), [**Michael Elad**](https://elad.cs.technion.ac.il/)

Abstract: The success of deep learning has been due, in no small part, to the availability of large annotated datasets. Thus, a major bottleneck in current learning pipelines is the time-consuming human annotation of data. In scenarios where such input-output pairs cannot be collected, simulation is often used instead, leading to a domain-shift between synthesized and real-world data. This work offers an unsupervised alternative that relies on the availability of task-specific energy functions, replacing the generic supervised loss. Such energy functions are assumed to lead to the desired label as their minimizer given the input. The proposed approach, termed "Deep Energy", trains a Deep Neural Network (DNN) to approximate this minimization for any chosen input. Once trained, a simple and fast feed-forward computation provides the inferred label. This approach allows us to perform unsupervised training of DNNs with real-world inputs only, and without the need for manually-annotated labels, nor synthetically created data. "Deep Energy" is demonstrated in this paper on three different tasks -- seeded segmentation, image matting and single image dehazing -- exposing its generality and wide applicability. Our experiments show that the solution provided by the network is often much better in quality than the one obtained by a direct minimization of the energy function, suggesting an added regularization property in our scheme.

## Citations

Please cite these papers in your publications if this code helps your research:

```
@article{golts2018deep,
  title={Deep energy: Using energy functions for unsupervised training of dnns},
  author={Golts, Alona and Freedman, Daniel and Elad, Michael},
  journal={arXiv preprint arXiv:1805.12355},
  year={2018}
}
```

```
@article{golts2018unsupervised,
  title={Unsupervised Single Image Dehazing Using Dark Channel Prior Loss},
  author={Golts, Alona and Freedman, Daniel and Elad, Michael},
  journal={arXiv preprint arXiv:1812.07051},
  year={2018}
}
```

## Getting Started

This repository contains: 

- Simple test functions for each application: `Test_Dehaze.py`, `Test_Matte.py`, `Test_Seg.py`
- More elaborate test functions which recreate the numeric and qualitative results in our paper: `Create_Figs_Dehaze.py`, `Create_Figs_Matte.py`, `Create_Figs_Seg.py`
- Implementations of all energy functions: `Dehazing_Loss.py`, `Matting_Loss.py` and `Segmentation_Loss.py`
- Training function to train your own model: `Deep_Energy.py`
- Neural network model we use throughout all experiments: `Models.py`
- Utilities: `train_utils.py`, `Utils.py`
- Configuration parameters file: `params.ini`

The training is performed with images saved in HDF5 files. These files can be downloaded via the following [link](https://www.dropbox.com/sh/nh5ps1ji7fovmrm/AACUCf4V9FvxOIHWpxbKv1NGa?dl=0). Once downloaded, place hdf5 files in the folder: "HDF5_files". To create the HDF5 yourself, please contact the owner for additional code. 

## Prerequisites

To perform test on saved models, you need the following:

- TensorFlow
- Numpy
- Scipy
- configargparse
- Matplotlib

To perform training using `Deep_Energy.py`, you will additionaly need:

- h5py

## Usage Example

To perform simple test, open `Test_Dehaze.py, Test_Seg.py, Test_Matte.py` in you IDE and run the scripts. Alternatively, from the `\Code` folder in the main repository, enter the following in the command:

`python Test_Dehaze.py`

`python Test_Seg.py`

`python Test_Matte.py` 

To run the extended evaluation scripts, open in your IDE and run, or using the following commands:

`python Create_Figs_Dehaze.py`

`python Create_Figs_Seg.py`

`python Create_Figs_Matte.py`

To run the training script, make sure you download beforehand the corresponsing HDF5_file, located in this [link](https://www.dropbox.com/sh/nh5ps1ji7fovmrm/AACUCf4V9FvxOIHWpxbKv1NGa?dl=0). Then, manually adjust the parameters in `params.ini` file, and run the training script (you can run it with the default values in `params.ini` as well) in the IDE or in the command line:

`python Deep_Energy.py`

## TODO

Create the HDF5 files instead of copying them.

## Saved Models

The model parameters used in our papers are located in the `\Results` folder, under each application. Some applications, for example single image dehazing contain several checkpoints: 27,30,33. For mild haze use 27 and for heavier amounts of haze use 30 and 33. 

When new training will be performed, other sub folders in `\Results` will be created with updated timestamps.

## Recreating Results in Papers

To recreate both numeric and qualitative results in our papers, use the following scripts: `Create_Figs_Dehaze.py`, `Create_Figs_Matte.py`, `Create_Figs_Seg.py`. Notice that this repository holds only few sample images (located in `\Datasets`) from the datasets we use to evaluate our work. To get exact results, one should download the entire datasets and place the images in the corresponding folders we provide, or change the links manually in the code. 

The datasets we use are all linked below:

#### Single Image Dehazing: 

HSTS, SOTS indoor and SOTS outdoor can be found in the following [link](https://sites.google.com/view/reside-dehaze-datasets/reside-v0). 

```
@article{li2018benchmarking,
  title={Benchmarking single-image dehazing and beyond},
  author={Li, Boyi and Ren, Wenqi and Fu, Dengpan and Tao, Dacheng and Feng, Dan and Zeng, Wenjun and Wang, Zhangyang},
  journal={IEEE Transactions on Image Processing},
  volume={28},
  number={1},
  pages={492--505},
  year={2018},
  publisher={IEEE}
}
```

Middlebury part of D-HAZY dataset can be found in the following [link](http://ancuti.meo.etc.upt.ro/D_Hazzy_ICIP2016/).

```
@inproceedings{ancuti2016d,
  title={D-hazy: A dataset to evaluate quantitatively dehazing algorithms},
  author={Ancuti, Cosmin and Ancuti, Codruta O and De Vleeschouwer, Christophe},
  booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
  pages={2226--2230},
  year={2016},
  organization={IEEE}
}
```

#### Image Matting:

We use 27 images from the train set in alphamatting.com dataset, which can be found [here](http://www.alphamatting.com/datasets.php). 

```
@inproceedings{rhemann2009perceptually,
  title={A perceptually motivated online benchmark for image matting},
  author={Rhemann, Christoph and Rother, Carsten and Wang, Jue and Gelautz, Margrit and Kohli, Pushmeet and Rott, Pamela},
  booktitle={2009 IEEE Conference on Computer Vision and Pattern Recognition},
  pages={1826--1833},
  year={2009},
  organization={IEEE}
}
```

#### Seeded Segmentation:

We use the Pascal VOC 2012 'val' dataset, which can be downloaded [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).

```
@article{everingham2010pascal,
  title={The pascal visual object classes (voc) challenge},
  author={Everingham, Mark and Van Gool, Luc and Williams, Christopher KI and Winn, John and Zisserman, Andrew},
  journal={International journal of computer vision},
  volume={88},
  number={2},
  pages={303--338},
  year={2010},
  publisher={Springer}
}
```

## Energy Function Sources

This code contains the implementation of deep energy for the following three applications: seeded segmentation, image matting and single image dehazing. The three energy functions with which we train our network are based on the following papers:

[1] Seeded Segmentation: 

```
@article{grady2006random,
  title={Random walks for image segmentation},
  author={Grady, Leo},
  journal={IEEE Transactions on Pattern Analysis \& Machine Intelligence},
  number={11},
  pages={1768--1783},
  year={2006},
  publisher={IEEE}
}
```

[2] Image Matting:

```
@article{levin2007closed,
  title={A closed-form solution to natural image matting},
  author={Levin, Anat and Lischinski, Dani and Weiss, Yair},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={30},
  number={2},
  pages={228--242},
  year={2007},
  publisher={IEEE}
}
```

[3] Single Image Dehazing:

```
@article{he2010single,
  title={Single image haze removal using dark channel prior},
  author={He, Kaiming and Sun, Jian and Tang, Xiaoou},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={33},
  number={12},
  pages={2341--2353},
  year={2010},
  publisher={IEEE}
}
```