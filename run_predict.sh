DEFAULT_PATH=/data/
FASTA_PATH=$DEFAULT_PATH/fasta/
A3m_PATH=$DEFAULT_PATH/a3m/
DCA_PATH=$DEFAULT_PATH/dca/
PSSM_PATH=$DEFAULT_PATH/pssm/
OUTPUT_PATH=$DEFAULT_PATH/prediction/
EXPERIMENT_PATH=./experiment/cross_entropy
DATABASE_PATH=$DEFAULT_PATH/uniclust30/uniclust30_2018_08
LENGTH_MIN=10
LENGTH_MAX=300
MSA_DEPTH_MIN=10
MSA_DEPTH_MAX=200

python ./predict.py \
-e=$EXPERIMENT_PATH \
--fasta_path=$FASTA_PATH \
--a3m_path=$A3m_PATH \
--dca_path=$DCA_PATH \
--pssm_path=$PSSM_PATH \
--database_path=$DATABASE_PATH \
--output_path=$OUTPUT_PATH \
--experiment_folder=$EXPERIMENT_PATH \
--length_min=$LENGTH_MIN \
--length_max=$LENGTH_MAX \
--msa_depth_min=$MSA_DEPTH_MIN \
--msa_depth_max=$MSA_DEPTH_MAX
