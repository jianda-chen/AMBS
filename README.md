# Learning Generalizable Representations for Reinforcement Learning via Adaptive Meta-learner of Behavioral Similarities
In this repository, we provide the code for the paper AMBS. If you are going to use the codes for your research, please cite our paper:
```
@inproceedings{
chen2022learning,
title={Learning Generalizable Representations for Reinforcement Learning via Adaptive Meta-learner of Behavioral Similarities},
author={Jianda Chen and Sinno Pan},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=zBOI9LFpESK}
}
```
# Dependencies
* Python >= 3.7
* PyTorch >= 1.7
* DeepMind Control Suite. Please refer to [dm_control](url:https://github.com/deepmind/dm_control) page to install the dependencies for DeepMind Control Suite
* Other python packages. Install other python dependencies in conda environment by the following command:
```
conda env update -f conda_env.yaml
``` 
# Getting Started
You can run the example by:
```
./run_local.sh
```
This script will save the experiment records in ```../log``` directory.

If you need to change the saving directory or run other tasks, please revise the file ```run_local.sh``` accordingly.

```
conda env create -f conda_env.yml
```
## License
This work is under [CC-BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/).
