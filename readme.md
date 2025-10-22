## Who You Are Matters: Bridging Topics and Social Roles via LLM-Enhanced Logical Recommendation


### Requirements

* Python 3.7+
* PyTorch 1.12+
* CUDA 11.6+


## Installation
RecBole works with the following operating systems:

* Linux
* Windows 10
* macOS X

RecBole requires Python version 3.7 or later.

### Install from conda

```bash
conda install -c aibox recbole
```

### Install from pip

```bash
pip install recbole
```

### Install from source
```bash
git clone https://github.com/RUCAIBox/RecBole.git && cd RecBole
pip install -e . --verbose
```

### Run
With the source code, you can use the provided script for initial usage of our method:

```bash
python run_tagcf.py --dataset=amazon_movie
```

This script will run the LogicRec model on the Amazon Movie dataset.

If you want to change the parameters, such as learning_rate, embedding_size, just set the additional command parameters as you need:

```bash
python run_tagcf.py --learning_rate=0.0001 --embedding_size=128
```

If you want to change the models, just run the script by setting additional command parameters:
```bash
python run_tagcf.py --model=[model_name]
```

#### Dataset
The [link](https://drive.google.com/drive/folders/14rS0lg7YaQksd1dLCU24PSl2_8_H7bgZ?usp=sharing) to our dataset is currently under construction and will be available upon publication. 