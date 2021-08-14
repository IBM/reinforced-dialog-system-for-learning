# Talk, do not read

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
cd ../Talk_/data/WoW-raw
wget http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz
tar zxvf wizard_of_wikipedia.tgz 
mv valid_random_split.json dev.json
```

You may continue to build two folders under the Talk_ directory
```shell script
cd ../Talk_
mkdir -p za/args
mkdir saved_models
mkdir logs
```

The downloaded dataset may miss some information, please refer to
```shell script
scripts/prepare_data/load_wikipedia_into_mysql.py 
```
to build up a Mysql database for Wikipedia (Please revise the code to fit your mysql setting)

When building the dataset using scripts in *scripts/prepare_data/prepare_wow_wiz_app*, the script would utilized the Wikipedia database to fill up the missing information

Use the following script to prepare data to train the wizard model and the apprentice model
```shell script
python scripts/prepare_wow_wiz_app/prepare_wow_1.1.py
```
(Difference between 1.1 and 1.6: Entire document as input to wizard - 1.1; Single sentence as input to wizard - 1.6)

## Train models
To train the wizard model, run the following command line

```shell script
python shell/train/train_wiz_1.1.sh
```

To train the apprentice model, run the following command line

```shell script
python shell/train/train_app_1.1.sh
```

To fine-tune the selector model using RL, run the following command line

```shell script
python shell/train/rl_self_play_1.5.sh
```

## Demo
Some self-play demos could be found in the folder *./demos* 


