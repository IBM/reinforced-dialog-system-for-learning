# Talk_Mini

This is the repo for using self-play and reinforcement learning to train a dialogue agent.

## Install the environment
We recommend creating a CONDA environment by
```bash
conda env create -f conda_env.yml
```

## Data preparation
Please prepare the data in another directory (you may name it *Talk_* ) under the same 
```shell script
mkdir -p ../Talk_/data/WoW-raw
```
Use [This script](https://github.com/facebookresearch/ParlAI/blob/master/parlai/tasks/wizard_of_wikipedia/build.py) to download the dataset to the above folder

Use the following script to prepare data to train the wizard model and the apprentice model
```shell script
python scripts/prepare_wow_wiz_app/prepare_wow_1.1.py
```
(Difference between 1.1 and 1.6: Entire document as input to wizard - 1.1; Single sentence as input to wizard - 1.6)

## Train models
To train the wizard model, run the following command line

```shell script
python shell/train/train_app_1.1.sh
```

To train the apprentice model, run the following command line

```shell script
python shell/train/train_app_1.1.sh
```

To fine-tune the selector model using RL, run the following command line

```shell script
python shell/train/rl_self_play_1.5.sh
```





