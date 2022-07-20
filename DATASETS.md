# How to install datasets

We suggest putting all datasets under the same folder (say `$DATA`) to ease management and following the instructions below to organize datasets to avoid modifying the source code. The file structure looks like

```
$DATA/
|–– imagenet/
|–– .../
|–– .../
|–– .../
```

Datasets list:
- [ImageNet](#imagenet)

### ImageNet
- Create a folder named `imagenet/` under `$DATA`.
- Create `images/` under `imagenet/`.
- Download the dataset from the [official website](https://image-net.org/index.php) and extract the training and validation sets to `$DATA/imagenet/images`. The directory structure should look like
```
imagenet/
|–– images/
|   |–– train/ # contains 1,000 folders like n01440764, n01443537, etc.
|   |–– val/   # contains 1,000 folders like n01440764, n01443537, etc.
```
- Download the `classnames.txt` to `$DATA/imagenet/` from this [link](https://drive.google.com/file/d/1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF/view?usp=sharing). The class names are copied from [CLIP](https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb).