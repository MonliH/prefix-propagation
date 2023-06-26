# prefix-propagation

Source code for ["Prefix-Propagation: Parameter-Efficient Tuning for Long Sequences"](https://arxiv.org/abs/2305.12086) (to be published at ACL 2023 main conference). Codebase is based off [P-tuning-v2](https://github.com/THUDM/P-tuning-v2).

![Prefix-Propagation Architecture](./figures/arch.png)

## Setup

We recommend creating a new [conda](https://docs.conda.io/en/latest/) enviornment for this project. To do so, run the following commands:
```bash
conda create --name prop python=3.10
conda activate prop
```

Then, install torch with conda:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

Finally, install the rest of the requirements:
```bash
pip install -r requirements.txt
```

## Usage

Training for prefix propagation can be done through the `run.py` script (make sure you have the conda environment activated). For easier configuration and easy grid hyperparameter search, see scripts in [`run_script`](./run_script/). For example, to train prefix-propgation with hyperparamter search on 20-newsgroups:
```bash
bash run_script/newsgroups_longformer_propagation.sh
```

### Wikihop Dataset Preparation

To use WikiHop, download data from [http://qangaroo.cs.ucl.ac.uk/](http://qangaroo.cs.ucl.ac.uk/) (can be done from command line with `gdown 1ytVZ4AhubFDOEL7o7XrIRIyhU8g9wvKA`). Then, unzip into `./data/`.