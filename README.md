This module only predicts the distance map using the direct coupling analysis (DCA) that is
computed by multiple sequence alignment (MSA).
Actually it is not recent work, we just want to study related work.
So, we don't have any plan of improvement. 

# Usage
## [Installation](https://github.com/spydr1/Protein-Resarch/blob/main/INSTALL.md)

## Predict
```angular2html

# prepare the database
put the database in [your data path]

# run container
nvidia-docker run -it --network=host -v /mnt/share/ssh:/root/.ssh -v [your data path]:/data pharmcadd:1.0

vim run_predict.sh

1. set your fasta folder
If you just want to predict, you need only fasta file.
please set your fasta folder and put your fasta file in folder.

2. set your database folder

sh run_predict.sh
```

## Train
### Create data
```
0. Crawling casp data (casp 12,13,14)
python crawling.py 
--download_path=[folder you want to save cif file]

1. Making the distance map (If you have cif file, you can skip thie procedure.)
python create_dsmap.py 
--cif_path=[cif folder] 
--dsmap_path=[distance map folder that you want to save] 
--fasta_path=[fasta folder that you want to save]

2. Making the a3m file
python create_a3m.py 
--fasta_path=[fasta folder that have the fasta file] 
--a3m_path=[a3m folder that you want to save] 
--database_path=[database folder]/[database name] - please check example.
--cpu=[number of cpu core that you want to use]

ex) 
--database_path=/J/data/uniclust30_2018_08/uniclust30_2018_08

3. Making the DCA, PSSM (tfrecord for training or numpy for prediction)
python create_dataset.py 
--a3m_path = [set your a3m path] (mandatory) 
--prpc_file = [dunbrack list] (option) 
--length_max = [maximum length of target fasta] (option, default : 300) 
--length_min = [minimum length of target fasta] (option, default : 50) 
--msa_rank_min = [minimum number of msa result] (option, default : 100) 
--msa_rank_max = [maximum number of msa result] (option, default : 200) 
--disable_gpu (option, not using recommened) 

tfrecord 
--tfrecord_path = [set your output path]
--dsmap_path = [set your distance map path]
or 
numpy 
--dca_path = [set your dca path]
--pssm_path = [set your pssm path]

ex)
cd /presearch_trrosetta/prepare/
1) tfrecord mode
python create_dataset.py --a3m_path=/data/a3m 
tfrecord --tfrecord_path=/data/tfrecord --dsmap_path=/data/dsmap  

2) numpy mode
python create_dataset.py --a3m_path=/data/a3m 
numpy --dca_path =/data/dca --pssm_path =/data/pssm

```

### Train
```angular2html
you need to set your training data path and validation data path.
This model need big memory, so please set carefully set batchsize.
one batch need about 7gb memory.

vim Pharmcadd/presearch_trrosetta/utils/config.py

set your train_path, valid_path, eval_path and hyper-parameter
if you want to only predict, you don't have to set path.

python train.py -e=[your experiment folder]

ex) python train.py -e=presearch_trrosetta/experiment/cross_entropy
```




