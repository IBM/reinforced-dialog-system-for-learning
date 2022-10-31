# Reinforced Dialog System For Learning

This is the repo for the NAACL 2022 paper  **[Learning as Conversation: Dialogue Systems Reinforced for Information Acquisition]()**.


## Build the environment
We recommend creating a CONDA environment by
```bash
conda env create -f conda_env.yml
```

Also, following [This Github Repo](https://github.com/shrimai/Focused-Attention-Improves-Document-Grounded-Generation), You have to copy the patch provided in patch folder to the desired location, i.e. find out the path where the transformers library is installed and replace the original ```generation_utils.py``` file in the transformers library with the ```patch/generation_utils.py``` file.

[comment]: <> (&#40;Pengshan has build a virtual env called **wow** on his softlayer machine *cai.sl.cloud9.ibm.com*, you may use that directly&#41;)

You may choose to download the preprocessed datasets, or build it yourself from scratch. 

### Download preprocessed datasets
<ul>
  <li><a href="https://ibm.box.com/s/katnocg311pxydz1bi0e6f58kcrwft8j">Datasets for pre-tuning the teacher bot</a></li>
  <li><a href="https://ibm.box.com/s/gp0gvj0bwhcqoe3r2o8vtvyjn7tb4m5r">Datasets for pre-tuning the student bot</a></li>
  <li><a href="https://ibm.box.com/s/bu2i2twsgvzcm1w6vdq5xvwt6gzjr0m8">Datasets for training/evaluating the coherence evaluation model</a></li>
  <li><a href="https://ibm.box.com/s/0l3gztvtqcdswo0u0s8s99n1ydnh46zs">Datasets for fine-tuning the teacher bot on Wikipedia</a></li>
  <li><a href="https://ibm.box.com/s/wpum5nnr8874do31z4ec2ze341ma335k">Datasets for fine-tuning the teacher bot on CNN-DailyMail</a></li>
  <li><a href="https://ibm.box.com/s/twqwms3h1u8m1t8gmvuvwaey957gomfb">Datasets for fine-tuning the teacher bot on Paper Abstracts</a></li>
</ul>

###Process the datasets yourself from scratch

Please prepare the data in another directory (e.g. you may name it *Talk_* ) under the same parent directory
```shell script
mkdir -p ../Talk_/data/WoW-raw
cd ../Talk_/data/WoW-raw
wget http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz
tar zxvf wizard_of_wikipedia.tgz 
mv valid_random_split.json dev.json
```
These commands will download and decompress the ***Wizard of Wikipedia*** dataaset which would be used to pre-tune our teacher and student bots. 

You may continue to build other folders under the Talk_ directory, which would be used to save hyper-parameters, model dumps and logs. 
```shell script
cd ../Talk_
mkdir -p za/args
mkdir saved_models
mkdir logs
```

The downloaded ***Wizard of Wikipedia*** dataset may miss some information, please refer to
```shell script
scripts/prepare_data/load_wikipedia_into_mysql.py 
```
to build up a Mysql database for Wikipedia (Please revise the code to fit your mysql setting)

When building the dataset using scripts in *scripts/prepare_data/prepare_wow_wiz_app*, the script would utilized the Wikipedia database to fill up the missing information

Use the following script to prepare data to pre-tune the wizard model and the apprentice model
```shell script
python scripts/prepare_data/prepare_wow_wiz_app/prepare_wow_1.1.py
```

To build datasets for RL piloted fine-tune (Wikipedia, CNN-DailyMail, Paper Abstracts), please refer to scripts in the following folder:

```````shell script
scripts/prepare_data/prepare_finetune_datasets
```````

To build coherence evaluation datasets (WoW-coherence), run the following python file:

```````shell script
python shell/prepare_data/prepare_coh-1.5.py
```````


[comment]: <> (&#40;Difference between 1.1 and 1.6: Entire document as input to wizard - 1.1; Single sentence as input to wizard - 1.6&#41;)

### Our pre-trained and fine-tuned model dumps

<ul>
  <li><a href="https://ibm.box.com/s/0upu1mhnrw9c0pzwl6el4vcjtsqmlx1l">Teacher bot pre-tuned on WoW dataset</a></li>
  <li><a href="https://ibm.box.com/s/4um9kvsdy9x3qiebwqdfc1vosknj2z5h">Student bot pre-tuned on WoW dataset</a></li>
  <li><a href="https://ibm.box.com/s/tl6m25uaur0yrgh9mch84k42csjsgz2c">Coherence evaluation model pre-traind on WoW-coherence dataset</a></li>
  <li><a href="https://ibm.box.com/s/kde2jht9970zzmlu1feecpt3mukfdczg">Teacher bot fine-tuned on Wikipedia</a></li>
  <li><a href="https://ibm.box.com/s/w0garm1sxsf3096njsbc6fop62dq8nch">Teacher bot fine-tuned on CNN-DailyMail</a></li>
  <li><a href="https://ibm.box.com/s/vzwidkjopkqac0xsulp5is9w8u4twvf0">Teacher bot fine-tuned on Paper Abstracts</a></li>
</ul>

You may train your own models following two-phase procedures:

### Phrase 1: Pre-tune

To pre-tune the wizard model, run the following command line:

```shell script
python shell/train_wiz.sh
```

To pre-tune the apprentice model, run the following command line:

```shell script
python shell/train_app.sh
```

To train the coherence evaluation model on WoW-coherence dataset, run the following command line:

```shell script
python shell/train_coh.sh
```

### Phrase 2: RL piloted fine tuning
To fine-tune the selector model using RL, run the following command line:

```shell script
python shell/rl_self_play.sh
```

You may revise the ```train_file_rl``` and ```validation_file_rl``` parameters to selects fine-tune datasets. 

The fine-tuning could take to up to two days on a songle A-100 GPU. 

[comment]: <> (Some self-play demos could be found in the folder *./demos* )


