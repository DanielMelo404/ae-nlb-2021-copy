# AE Studio - NLB 2021
This repo is dedicated to reproducing [AE Studio](https://ae.studio/brain-computer-interface)'s winning submissions to the [Neural Latents Benchmark Challenge 2021](https://neurallatents.github.io/).  Note that code included here is intended to be the minimum needed to reproduce those submissions results.  We will be releasing a write-up of our approach and process, including other experimental paths we tried that are not contained in this code base and did not contribute directly to our winning submissions.

This is a fork of [neural-data-transformers](https://github.com/snel-repo/neural-data-transformers), which was the basis for the winning submissions.

For details on the approach we took to the challenge, please see [Our Approach](approach.md).

### AE Contributors
* Darin Erat Sleiter [@dsleiter](https://github.com/dsleiter)
* Joshua Schoenfield [@joshuaAtAE](https://github.com/joshuaAtAE)
* Mike Vaiana [@vaiana](https://github.com/vaiana)

### Acknowledgements

**Thanks to the organizers** for putting this competition together, we had a blast competing!

Many thanks to Sumner L Norman ([@sumner15](https://github.com/sumner15)) for his guidance and advice, especially in the areas of neuroscience and neural decoding.

## Model Description

We experimented with many different types of models, but our winning submission was an ensemble of NDT models that were trained using Bayesian hyperparameter optimization. The focus of this repo is on reproducing those model ensembles.

**AESMTE3 = AE Studio Model: Transformer Ensemble 3**

To build the ensemble, we ranked the candidate model snapshots produced via bayesopt by the validation co-bps, and then ensembled N top models by taking the mean of predicted rates.

The number of models ensembled for each dataset were:

| Dataset        |  Ensemble Size |
| -------------- | -------------- |
| mc_maze        |              8 |
| mc_rtt         |             13 |
| area2_bump     |             21 |
| dmfc_rsg       |             13 |
| mc_maze_large  |              8 |
| mc_maze_medium |              8 |
| mc_maze_small  |              7 |

The ensemble size was chosen by first ordering individual models by validation co-bps and then ensembling the top N models for N=1...M where M was some reasonably large number (30-50).  The final choice of ensemble size was the N that maximized validation co-bps.  Due to a bug only discovered after submission, the N used for the actual submission was slightly lower than the optimal N for a few of the datasets.  However, we also observed that the co-bps was not overly sensitive to the exact value of N, so this likely did not have a large effect.

We have made all our trained NDT model checkpoints from bayesopt available for download at: https://zenodo.org/record/5875246#.YefGXYRByEI (which can also be downloaded via the `download_checkpoints.py` script).

This archive also includes details on the commands used to train all of the models, as well as csvs containing the hyperparameters and validation metric scores for each of the model checkpoints. See the README in the archive for more details.

For training, we used machines with 4 GPUs. The training time for the full Bayesian hyperparameter optimization of each dataset was:

| Dataset        |  Total Training Time | # of Models Trained |
| -------------- | -------------------- | ------------------- |
| mc_maze        |             43.6 hrs |                 120 |
| mc_rtt         |             14.6 hrs |                 120 |
| area2_bump     |              5.9 hrs |                 100 |
| dmfc_rsg       |             47.8 hrs |                 130 |
| mc_maze_large  |              2.7 hrs |                 120 |
| mc_maze_medium |              6.1 hrs |                 120 |
| mc_maze_small  |             13.2 hrs |                 120 |

The time to compute ensembles was negligible.

## Validation Quickstart

Assuming you already have an environment with the following:
* python 3.8+
* all requirements in `requirements.txt` installed
* the command line utilities `wget` and `unzip` installed
* at least 30GB of free disk space
* at least 32GB RAM (all datasets except `mc_maze` can be validated on 16GB)

If these requirements are not yet met, see [Machine Setup](#machine-setup) for setting up the machine (and free up disk space if needed). Note that you don't need a GPU to run the validation. Once the machine is setup, continue with the following steps.

1. download the NLB DANDI files:
    ```
    python download_data.py
    ```
    If you already have the files downloaded, you can set the environment variable `NLB_DATA_RAW` to the parent directory containing the numbered DANDI datasets (see [Environment Variables](#environment-variables) below).

2. download our model checkpoints:
    ```
    python download_checkpoints.py
    ```
    If you have already downloaded and extracted the checkpoints, you can set the environment variable `SUBMISSION_VALIDATION_ROOT` to the root directory containing the dataset directories with the checkpoints (see [Environment Variables](#environment-variables) below).

3. download the AESMTE3 test submission files from EvalAI to a local directory, taking note of which submission file has rates for which datasets.

4. validate the submitted test rates for each dataset by re-compiling the model ensemble rates by inferring from the model checkpoints and compare against the submitted rates:
    ```
    python validate_test_submission.py {dataset name} {path to submission file}
    ```
    example:
    ```
    python validate_test_submission.py mc_maze mc_maze_submission.h5
    ```
    This script will recompile the ensemble test rates and compare the numpy arrays against the submission. See the script for more details.


## Machine Setup
1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html).  Skip this step if minconda or anaconda are already installed.
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b 
source ~/.bashrc
```

2. Clone this repository
    ```
    git clone https://github.com/agencyenterprise/ae-nlb-2021.git
    ```

3. Clone nlb_tools
    ```
    git clone https://github.com/neurallatents/nlb_tools.git
    ```

4. Create a virtual environment
    ```
    conda create --name ae-nlb-2021 python=3.8 && \
    conda activate ae-nlb-2021
    ```

5. Install nlb_tools
    ```
    pip install -e nlb_tools/
    ```

6. Install requirements
    ```
    cd ae-nlb-2021
    pip install -r requirements.txt
    ```

7. Install needed commmand line utilities
    The download script requires the command line utilities `unzip` and `wget` to be installed if they are not already. If you are using Ubuntu, if `wget` is not already installed, you can install it using the command
    ```
    sudo apt install wget
    ```
    Similarly, you can install `unzip` using
    ```
    sudo apt install unzip
    ```

## Data setup
There are several data files that are created when using this repo.  The paths to these data files can be set explicitly with environment variables but default paths are shown below.  Note the "main" path is set with `NLB_HOME`, and all data will be saved into this folder.  If you want to change any individual path you can do that 
by explicitly setting the corresponding environment variable.  Note that none of the directories corresponding to environment variables need to exist they will be created.


### Environment variables
* `NLB_HOME` All data, configs, and checkpoints are saved in this directory. Default is `$HOME/nlb_2021`
* `NLB_DATA_RAW` The raw `.nwb` dandiset files for each experiment in the challenge will be saved here.  Default is `$NLB_HOME/raw`.
* `NLB_CACHE_DIR` Cache dir for internal preprocessing steps.  Default is `$NLB_HOME/processed`
* `NDT_CACHE_DIR` Cache directory for NDT specific training and validation datasets.  Default is `$NLB_HOME/neural-data-transformer`
* `RAY_TUNE_HOME` All Ray Tune hyperopt results will be saved here. Default is 
`$NLB_HOME/ray_results/neural-data-transformer`
* `CHECKPOINT_DIR` Directory for saving single model run checkpoints. This is unused during hyperopt. Default is `$NLB_HOME/checkpoints`
* `CONFIG_DIR` Directory for saving model and hyperopt config files.  See the steps below for automatically generating configs from the included templates.  Default is `$NLB_HOME/configs`
* `SUBMISSION_VALIDATION_ROOT` Root directory for the model checkpoints for validation of AESMTE3. Default is `$NLB_HOME/submission-validation`


### Data Steps

1. Download Data \
    This step downloads a relatively large amount of data and may take a while.  If you already have the data on your machine, set the environment variable `$NLB_DATA_RAW` to the top level directory containing the raw dandiset data so that the data will not be re-downloaded.
    ```
    python download_data.py
    ```

2. Generate Configs \
    This step will generate the configuration files needed to run hyperopt.  It will create configs which reflect the correct paths based on the environment variables (or their defaults) above.
    ```
    python generate_all_configs.py
    ```

You should now have all the data and configurations you need to run hyperopt.


----
## Model Training
We used Bayesian hyperparameter optimization to find the models used in our ensemble. To reproduce our optimization hyperparameter search use:
```
python run_hyperopt.py <dataset_name> [--seed=42] [--samples=100]
```
For example to run `mc_maze_small` with 50 samples (and default seed of 42)
```
python run_hyperopt.py mc_maze_small --samples 50
```

This will launch a Ray Tune hyperopt search using the scikit-optimize implementation of Bayesian optimization. This can take a while depending on the dataset, the number of samples, and the compute power of your machine.  If you have a terminal attached to an cloud compute instance we recommend running in a detached [screen](https://www.gnu.org/software/screen/manual/screen.html) to ensure the run isn't interrupted by a broken pipe to the instance.

```
screen 
conda activate ndt
python run_hyperopt.py <dataset_name>
```
Then ctrl+a d to detach the screen.  


### GPU Setup
The model training script assumes you have GPUs available on your machine together with the correct drivers.
We recommend using the [lambda stack](https://lambdalabs.com/lambda-stack-deep-learning-software#server-installation) provided by [lambda labs](https://lambdalabs.com/).  For convenience the lambda stack script is copied below.

```
LAMBDA_REPO=$(mktemp) && \
wget -O${LAMBDA_REPO} https://lambdalabs.com/static/misc/lambda-stack-repo.deb && \
sudo dpkg -i ${LAMBDA_REPO} && rm -f ${LAMBDA_REPO} && \
sudo apt-get update && \
sudo apt-get --yes upgrade && \
echo "cudnn cudnn/license_preseed select ACCEPT" | sudo debconf-set-selections && \
sudo apt-get install --yes --no-install-recommends lambda-server && \
sudo apt-get install --yes --no-install-recommends nvidia-470 libcuda1-470 nvidia-opencl-icd-470 && \
sudo apt-get install --yes --no-install-recommends lambda-stack-cuda
```
**Note** If you are running on EC2, you may need to stop and restart your instance at this point, otherwise you may need to reset your machine.





