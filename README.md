# seq_simul
seq_simul is a DNA storage channel simulation tool.

## Download
Clone repo through ssh

```
git clone ssh://git@github.com/albert-no/seq_simul.git
```

## Requirements
### 1. editdistance
Install `editdistance` package

```
conda install -c conda-forge editdistance
```

### 2. seqanpy
Install [seqanpy](https://github.com/iosonofabio/seqanpy) package for alignment

1. download seqan 2.4.0 ([Download_link](http://packages.seqan.de/))
2. unzip the file and export the `include` folder path
3. download `swig` and export swig
4. install seqanpy using pip

```
export SEQAN_INCLUDE_PATH=/PATH/TO/SEQAN/LIBRARY/seqan-library-2.4.0/include
sudo apt install swig
export SWIG=swig
pip install seqanpy
```

## Setup
`seq_simul` is implemented using Python 3.7 and should be installed before running.

```
pip install -e .
```

## Generate Dataset (Processing Pipeline)
Processing pipeline generates the data for training simulator.

### Input and Output of Processing Pipeline
The inputs of the processing pipeline are oligo text file (`oligos.txt`)
which contains the oligo (encoded data)
and FASTQ file (`.fastq`) which contains the read and quality-score.
The output of the processing pipeine is `data` files.
The `data` file stores the matching result.
Each matching result corresponds to 5 lines in `data` file.

- read from fastq
- qscore (that corresponds to read)
- matched oligo
- edit-distance between the read and the oligo
- index of matched oligo

### Processing Pipeline Overview
The processing pipeline does the following:

1. Convert FASTQ file into reads file (extract reads and qscores from FASTQ file)
2. Split reads file (split read files into subread files)
3. Find edit distance and quality value between split files and oligo text file
4. Save results in data file
5. Separate data files into train-set and test-set under the defined proportion
6. Separate data files in train-set and test-set through edit-distance (0 edit-distance & defined edit-distance range)
7. Modify and save the data file which is saved in non-zero edit-distance folder into three errors data files (insertion, deletion, substitution data file)

#### 1. Matching oligos and reads
To run the `processing.py`, parameters in `main` have to be modified appropriately.
1. `path` is a path to FASTQ (.fastq) and oligo text (.txt) file
2. `fname` is a name of FASTQ and oligo text file.
    For example, `fname=constraint` implies that FASTQ file name is `constraint.fastq` and oligo text file name
    is `constraint.txt`.
3. `edit_distance_limit` is a maximum allowable edit-distance between the read and the matched oligo.
4. `split_num` is a number of read which is saved in single reads file.

Then, run `processing.py` at root to generate data files.

#### 2. Get Train, Test Dataset
Parameters of `get_train_test_set.py` need to be modified as personal use.
1. `folder_path` is a path of matched dataset.
2. `division_ratio` is a division proportion of the train-set and test-set. (if `division_ratio=0.8`, it return 80% of train-set and 20% of test-set)

Run `get_train_test_set.py` returns the proportionally divided train and test folder. 

#### 3. Split Train, Test Dataset within Edit Distance & Split Three Error Data
To customize parameters of `trim_and_split.py` as follows.
It trims the length of the sequences and save separately within edit distance.
Also, split data into three errors(insertion, substitution, deletion).
Trimmed data are saved in the train, test folders, and error contained data are saved in insertion, deletion, substitution folders which are generated in the train and test folders.
Additionally, for the profile vector that is essential for simulation, proportions of the errors are saved as a log file named `data_proportions.log`.

1. `folder_path` is a path of matched  data.
2. `min_edit_distance` & `max_edit_distance` are the range of the edit distance to save separately from zero edit distance.
3. `limit_length` is the maximum length of the sequence to save.

Run `trim_and_split.py`

## Train the Model (Training Pipeline)
Basically, simulator is composed of three read generator(insertion, substitution, deletion), and qscore generator. 
GANs are trained with GRU generator and 1D-CNN discriminator, except substitution which is trained with Transformer generator.
The training pipeline trains multiple models based on arguments.

### Arguments
Model parameters such as number of layer in the Generator (`G_num_layer`) and the Discriminator (`D_num_layer`),
and the size of hidden layer of the Generator (`G_hidden_size`) and Discriminator (`D_hidden_size`) can be tuned.
For transformer generator, number of head (`G_num_head`) and positional drop out (`pos_drop_prob`) also can be tuned.
Training parameters such as critic (`G_critic`), lambda(`lambda_gp`) for WGAN and learning rate for the Generator (`G_lr`) and Discriminator (`D_lr`),
decaying parameter for ADAM optimizer (`b1`, `b2`) can be tuned as well.
Detailed and defaulted arguments are at `seq_simul/data/default_args.json`.


### 1. Train GANs for Read Generator
There are two types of parameters:

1. fixed parameters that are fixed during the parameter sweep.
2. iterating parameters that varies over training.

Fixed parameters are set via `fixed_dict` in `read_sweep.py`
where keys are name of fixed parameters and values are corresponding fixed values.
Iterating parameters are set via `iter_dict`
where keys are iterating parameter and values are list of values
Then, `read_sweep.py` runs `wrapper` which is defined in `seq_simul/train/wrapper.py`
which trains the models for all possible combinations of iterating parameters.
Importatly, `read_only: True` and `qscore_only: False` should be stated in `fixed_dict`.

After setting parameters, run `read_sweep.py` with parameters to run the training pipeline.

```
python read_sweep.py
```


### 2. Train GANs for Qscore Genrator
Qscore-Generator also has `fixed_dict` for fixed parameters,
and `iter_dict` for iterating parameters.
Set those dictionaries accordingly,
then run `qscore_sweep.py` which runs `wrapper` in `seq/simul/train/wrapper.py`.
It trains the models for all possible combinations of iterating parameters.
Importatly, `read_only: False` and `qscore_only: True` should be stated in `fixed_dict`.

After setting parameters, run `qscore_sweep.py` with parameters to run the training pipeline.

```
python qscore_sweep.py
```

## Simulate
Simulator is composed of read-simulator and qscore-simulator.
Read-simulator also composed of insertion-generator, substitution-generator, and deletion-generator.
Based on profile-vector that saved in `error_proportion.log`, the input oligo enters the multiple or single generator among three generators.
Qscore-simulator gets the result of read-simulator as input and generate quality score sequences.
 
1. The input of the read-simulator is `.txt` oligo file, and outputs the `.data` file.
2. The input of the qscore-simulator is `.data` file, and outputs the `.fastq` file or `.data` file.

Your output will be a `.fastq` file when using qscore simulator with `mode=qscore_fastq`.
The format from top to bottom:
1. index
2. read sequence (simulated read)
3. \+
4. quality score (generated qscore)

Your output will be a `.data` file when using qscore simualtor with `mode=qscore_data`
1. index
2. read sequence (simulated read)
3. \+
4. quality score (generated qscore)
5. oligo


Some parameters are modified before simulating sequences.
Otherwise, model parameters are loaded from json files.
1. `error_proportion_file` : path of the error proportion log file
2. `simulation_fname` : path of input data
3. `simulated_result_path` : name of file saved in `results/simulations/`
3. `ins/sub/del/qscore_simulation_folder` : path of folder of trained insertion, substitution, deletion and qscore
4. `ins/sub/del/qscore_simulation_fname` : file name of trained insertion, substitution, deletion and qscore
5. `ins/sub/del/qscore_epoch_list` : epochs for generating sequences 

The above process is implemented in `seq_simul/simulator/seq_simulator.py`, run `simulator.py` to get the result.

```
python simulator.py
```

The result of simulation and the real expriment can be compared via statitstics tool.


## Statistics
There are 2 ways to analyze statistics.

### 1. Reads
- plot proportion of each index errors : all, insertion, substitution, deletion
- number of errors : all, insertion, substitution, deletion
- number of different base pair : insertion, substitution, deletion
- number of consecutive dashes : insertion, deletion
- plot consecutive dashes : insertion, deletion

### 2. Quality scores
- plot positional mean of quality scores.
- plot the distribution of error(insertion, substitution) occurred quality score.

1. `mode` : select mode for analyzing (all|read|qscore) 
2. `error_name` : error name to get statistics with 4-options(all|insertion|deletion|substitution).
3. `original_data_path` : folder path of real `.data` file 
4. `original_fname` : file name of real `.data`
5. `generated_result_path` : folder path of generated file
6. `generated_fname` : file name of generated file (`.data` or `.fastq`)
7. `read_padded_length` : sequence length with pad to load reads.
8. `qscore_padded_length` : sequence length with pad to load quality scores.

`python stats.py` returns the designated statistics and results are save at `results/statistics/`

```
python stat.py
```

## End-to-end example
Following direction, test the end-to-end example with mini dataset(`/test_constraint`).

1. Processing data
```
python processing.py --path=seq_simul/data/test_constraint/ --fname=test_constraint --split_num=25 --convert_fastq=True
```

2. Get train and test dataset
```
python get_train_test_set.py --folder_path=seq_simul/data/test_constraint/test_constraint/data/ --division_ratio=0.8
```

3. Get only error contained data and customized edit-distance data
```
python trim_and_split.py --folder_path=seq_simul/data/test_constraint/test_constraint/data --limit_length=145
```

4. Train read-GAN (insertion|substitution|deletion)
Customize fixed and iterative dictionary in `read_sweep.py`
insertion `data_path` : `seq_simul/data/test_constraint/test_constraint/data/train/edit_1_5/insertion/`
substitution `data_path` : `seq_simul/data/test_constraint/test_constraint/data/train/edit_1_5/substitution/`
deletion `data_path` : `seq_simul/data/test_constraint/test_constraint/data/train/edit_1_5/deletion/`
```
python read_sweep.py
```

5. Train qscore-GAN (errorness|error-free)
Customize fixed and iterative dictionary in `qscore_sweep.py`
errorness quality score `data_path` : `seq_simul/data/test_constraint/test_constraint/data/train/edit_1_5/`
error-free quality score `data_path` : `seq_simul/data/test_constraint/test_constraint/data/train/edit_0/`
```
python qscore_sweep.py
```

6. Simulate with trained read-generator and qscore-generator (need json and pth file)
   and designate multiple epochs for simulating
```
python simulator.py --mode=read
--error_proportion_file=seq_simul/data/test_constraint/test_constraint/error_proportion.log
--simulation_fname=seq_simul/data/test_constraint/test_constraint/data/test/edit_1_5/test_test_constraint_split_aa.data
--simulated_result_fname=read_simulated
--ins_simulation_folder=results/MMDDhhmm_sweep/trained_parameters
--sub_simulation_folder=results/MMDDhhmm_sweep/trained_parameters
--del_simulation_folder=results/MMDDhhmm_sweep/trained_parameters
--ins_simulation_fname=MMDDhhmm_{param}
--sub_simulation_fname=MMDDhhmm_{param}
--del_simulation_fname=MMDDhhmm_{param}
--ins_epoch_list n1 n2 n3 ..
--sub_epoch_list n1 n2 n3 ..
--del_epoch_list n1 n2 n3 ..

python simulator.py --mode=qscore_fastq
--simulation_fname=results/simulations/read_simulated.data
--simulated_result_fname=qscore_simulated
--qscore_simulation_folder=results/MMDDhhmm_sweep/trained_parameters
--qscore_simulation_fname=MMDDhhmm_{param}
--qscore_epoch_list n1 n2 n3 ...
```

7. get statistics with simulated data file
```
python stats.py --mode=all (or read|qscore) 
		--error_name=all (or insertion|deletion|substitution)
		--generated_result_path=results/
		--generated_fname=simulated.data
```

## Simulate Sequences with Trained Parameters
User can simulate sequences using parameter of pre-trained generators.(at the folder named `pre-trained_parameters`)

```
python simulator.py --mode=read
		    --ins_simulation_folder=pre-trained_parameters
		    --sub_simulation_folder=pre-trained_parameters
		    --del_simulation_folder=pre-trained_parameters
		    --ins_simulation_fname=insertion
		    --sub_simulation_fname=substitution
		    --del_simulation_fname=deletion
		    --ins_epoch_list 0 1 2 3 4 5 6 7 8 9
		    --sub_epoch_list 0 1 2 3 4 5 6 7 8 9
		    --del_epoch_list 0 1 2 3 4 5 6 7 8 9

python simulator.py --mode=qscore_fastq
		    --qscore_simulation_folder=pre-trained_parameters
		    --qscore_simulation_fname=errorness
		    --qscore_epoch_list 0
		    --simulation_fname=results/simulations/simulated.data
```

## Test
To run the unittest, use the following command at root.

```
python -m unittest
```
