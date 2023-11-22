# Partial Registration Network

## Prerequisites 
PyTorch=1.0.1: https://pytorch.org  (PyTorch 1.1 has a svd bug, which will crash the training)

scipy>=1.2 

numpy

h5py

tqdm

sklearn

## Conda environment 

conda env create -f environment.yml --name prnet

conda activate prnet

## Training

### exp1 modelnet40

python main.py --exp_name=exp1

### exp2 modelnet40 unseen

python main.py --exp_name=exp2 --unseen=True

## exp3 modelnet40 gaussian noise

python main.py --exp_name=exp3 --gaussian_noise=True

## Posmap generation usage
1. Clone the repository from face3d:
   ```
   git clone https://github.com/YadiraF/face3d
   cd face3d
   ```
2. Put the script Generate_posmap_300WLP.py under the /examples directory of face3d repo
3. Compile the c++ files and preparing BFM data as instructed in https://github.com/yfeng95/face3d
4. Download the 300W_LP dataset from https://drive.google.com/file/d/0B7OEHD3T4eCkVGs0TkhUWFN6N1k/view?usp=sharing&resourcekey=0-WT5tO4TOCbNZY6r6z6WmOA, after unzipping, put under examples/Data directory of face3d
5. Run dataset generation script:
   ```
   cd face3d/examples
   python Generate_posmap_300WLP.py
   ```
   The dataset generated will be saved under examples/results directory. 
## Citation
Please cite this paper if you want to use it in your work,

	@InProceedings{Wang_2019_NeurIPS,
	  title={PRNet: Self-Supervised Learning for Partial-to-Partial Registration},
	  author={Wang, Yue and Solomon, Justin M.},
	  booktitle = {33rd Conference on Neural Information Processing Systems (To appear)},
	  year={2019}
	}

## Code Reference

Code reference: Deep Closest Point

## License
MIT License
