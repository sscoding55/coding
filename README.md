# Coding

## Dependencies
- Python 3.10
- PyTorch 2.7.1
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

## Create environment and install packages
- `conda create -n Coding python=3.10`
- `conda activate Coding`
- `pip install -r requirements.txt`

## Testing

Download the data and log folders from [GoogleDrive](https://drive.google.com/drive/folders/1-Evu4mN3DfKGJTjoOs_SfRmhJg9uQ8LD?usp=drive_link) and place them in the project directory.

- Test the segmentation performance

  
`python test.py --device 0 --img_path data`





