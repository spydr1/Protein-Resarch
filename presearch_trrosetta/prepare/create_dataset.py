import pdb

import numpy as np
import os
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from presearch_trrosetta.prepare import utils_baker
from presearch_trrosetta.utils import vocab
import tqdm
import tensorflow as tf
from PIL import Image
import random
import logging

def _bytes_feature(value):
	"""Returns a bytes_list from a string / byte."""
	if isinstance(value, type(tf.constant(0))):
		value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
	"""Returns a float_list from a float / double."""
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
	"""Returns an int64_list from a bool / enum / int / uint."""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


# reference : https://www.tensorflow.org/tutorials/load_data/tfrecord
def serialize_example(fasta,
					  seq,
					  length,
					  f2d_dca,
					  f1d_pssm,
					  target,
					  num_a3m,) :
	"""
	return the serialized string.
	"""


	# Create a dictionary mapping the feature name to the tf.train.Example-compatible
	# data type.

	seq = tf.convert_to_tensor(seq,dtype=tf.int32)
	length = tf.convert_to_tensor(length,dtype=tf.int32)

	f2d_dca_tensor = tf.convert_to_tensor(f2d_dca,dtype=tf.float32)
	f2d_dca_result = tf.io.serialize_tensor(f2d_dca_tensor)

	f1d_pssm_tensor = tf.convert_to_tensor(f1d_pssm,dtype=tf.float32)
	f1d_pssm_result = tf.io.serialize_tensor(f1d_pssm_tensor)

	target_tensor = tf.convert_to_tensor(target,dtype=tf.int32)
	target_result = tf.io.serialize_tensor(target_tensor)

	num_a3m =tf.convert_to_tensor(num_a3m,dtype=tf.int32)

	feature = {
		'fasta': _bytes_feature(fasta.encode('ascii')),
		'seq': _int64_feature(seq),
		'length': _int64_feature([length]),
		'f2d_dca': _bytes_feature(f2d_dca_result),
		'f1d_pssm' :_bytes_feature(f1d_pssm_result),
		'target': _bytes_feature(target_result),
		'num_a3m' : _int64_feature([num_a3m])
	}
	# Create a Features message using tf.train.Example.

	example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
	return example_proto.SerializeToString()

def load_a3m(file : str) -> np.array:
	"""
	parsing the only sequence without name of fasta from a3m file
	"""
	assert 'a3m' in file, "only a3m file."
	a3m = np.loadtxt(file, dtype=str, delimiter='\n')
	# a3m has name line and AA line
	# ex) >fasta~~
	# ex) >MAB ~~
	a3m = a3m[1::2]
	return a3m

def cut_a3m(a3m : np.array,
			msa_depth_max : int) -> np.array:
	"""
	cut the a3m to corresponding msa_depth_max.
	"""
	# we use only {msa_depth_max} result of Msa
	# num_a3m = len(a3m)
	a3m = a3m[:msa_depth_max]
	for i, seq in enumerate(a3m):
		a3m[i] = seq[:len(a3m[0])]

	return a3m


def fasta_from_file(file):
	file_name = os.path.basename(file)
	fasta = os.path.splitext(file_name)[0]
	return fasta

def prpc_msa(a3m,
			 msa_depth_max = 200,
			 ) :

	"""
	Make the combined single dictionary from each feature that are fasta, seq, length, distance, f1d_pssm and f2d_dca.
	"""
	num_a3m = len(a3m)
	a3m = cut_a3m(a3m, msa_depth_max)
	a3m = np.array([list(map(lambda x: vocab.onehot_dict[x], seq.upper())) for seq in a3m], dtype=np.uint8)
	onehot_a3m  = tf.one_hot(a3m, 21, dtype=tf.float32)

	# baker DCA
	w = utils_baker.reweight(onehot_a3m, 0.8)
	f1d_pssm = utils_baker.msa2pssm(onehot_a3m[0, :, :20], w)

	f2d_dca = utils_baker.fast_dca(onehot_a3m, w)
	f1d_pssm = f1d_pssm

	return {'seq':a3m[0],
			'length': len(a3m[0]),
			'f2d_dca':f2d_dca,
			'f1d_pssm':f1d_pssm,
			'num_a3m' : num_a3m,
			}



def dunbrack_filtering(prpc_file ):
	"""
	parsing the only fasta name from dunbrack list.
	we can set parameter so it is changeable.

	dunbrack
    i) has template(s) not too far or close in sequence space;
    ii) does not have strong contacts to other protein chains,
    iii) should contain minimal fluctuating (i.e. missing density) regions.
    iv) sequence similarity 40%
    v) Minimum resolution 2.5 Å
    vi) limiting their size to 50-300 residues

    :return : list of fasta that included in dunbrack list, all of name format is "fasta_chain"
	"""

	# prpc_list has multiple column (fasta name ,length, ...)
	# parsing only fasta name.
	# In dunbrack list, naming format of fasta is  different
	# 2NW2A - > 2NW2_A ,We add '_' symbol name between chain
	prpc_list = np.loadtxt(prpc_file, dtype='str', skiprows=1)
	dunbrack_fasta, length, *_ = prpc_list
	dunbrack_fasta = [f'{fasta[:4]}_{fasta[4:]}' for fasta in dunbrack_fasta]
	return dunbrack_fasta

def save_data(a3m_path,
			  dsmap_path = None,
			  dca_path = None,
			  pssm_path = None,
			  tfrecord_path = None,
			  prpc_file = None,
			  length_min = 50,
			  length_max = 300,
			  msa_depth_min = 100,
			  msa_depth_max = 200,
			  mode = 'tfrecord'):
	"""
	Result of MSA is must calculated to fixed matrix.
	There are many method (potts model, co-variance matrix, co-occurrence matrix)
	We use the co-variance matrix from baker group, this time.
	And then, save the tfrecord file for training or split file each f2d_dca(.npy) and f1d_pssm(.txt) for predict.

	Args :
	:param a3m_path :
	:param dsmap_path :
	:param dca_path :
	:param pssm_path :
	:param tfrecord_path :
	:param prpc_file : http://dunbrack.fccc.edu/Guoli/pisces_download.php
	:param msa_depth_max: how many use msa result ?
	:param msa_depth_min:
	:param length_max: minimum threshold of seuqunce length
	:param length_min: maximum threshold of seuqunce length
	:param mode: tfrecord or split file
	
	"""
	if mode =='tfrecord':
		assert os.path.isdir(dsmap_path), "please fill the correct distance path," \
											 "if you want to predict, don't use tfrecord mode."


	a3m_list = [os.path.splitext(path)[0] for path in os.listdir(a3m_path)]
	if prpc_file:
		if os.path.isfile(prpc_file):
			logging.info("you are using dunbrack pre-processed file")
			# todo : considering the another pre-processed list
			dunbrack_list = dunbrack_filtering(prpc_file=prpc_file)
			a3m_list = np.intersect1d(a3m_list, dunbrack_list)
	else :
		logging.info("you are not using pre-processed file \n "
			  "We use all of data in data_path")

	# todo : considering the restriction of file name, now we use the format "{fasta}_{chain}.a3m, seq, ...
	# filtering
	# we exclude the very short and long case.
	# we exclude very small number of msa result case.
	filtered_list = []
	for fasta in tqdm.tqdm(a3m_list, desc="filtering about fasta length, number of msa..."):
		a3m = f'{a3m_path}/{fasta}.a3m'

		a3m = load_a3m(a3m)
		seq = a3m[0]
		if msa_depth_min <= len(a3m):
			if length_min <= len(seq) <= length_max:
				filtered_list.append(fasta)
	logging.info(f"number of filtered_list is {len(filtered_list)}")

	# shuffle.
	# 1. seed is fixed -> always same order.
	# 2. when use the large file, shuffling the order is hard because it must be load into the memory.
	# 	 so we have to shuffle before creating tfrecord.
	#    -> If you want to shuffle after creating tfrecord, you must have many RAM. (~500gb)

	random.seed(12345)
	random.shuffle(filtered_list)
	with open('./datalist.txt', 'w') as file_obj:
		file_obj.write('\n'.join(filtered_list))

	count = 0
	# Data file is very big, so we must split to multiple tfrecord file.
	# todo : thinking about the what is proper method to save data file.
	if mode== 'tfrecord':
		os.makedirs(f'{tfrecord_path}', exist_ok=True)
		writer_idx = 0
		tfrecord_file = f'{tfrecord_path}/data{writer_idx}.tfrecords'
		writer = tf.io.TFRecordWriter(tfrecord_file)
		for fasta in tqdm.tqdm(filtered_list):
			# length of residue is different, slicing to corresponding target.
			# load -> cut -> one-hot
			a3m = load_a3m(f'{a3m_path}/{fasta}.a3m')

			distance_file = f'{dsmap_path}/{fasta}.png'
			distance = np.array(Image.open(distance_file), np.int32)

			prpc_data = prpc_msa(a3m, msa_depth_max = msa_depth_max)
			prpc_data['fasta'] = fasta
			prpc_data['target'] = distance

			example = serialize_example(**prpc_data)
			writer.write(example)

			if (count+1)%1000==0:
				# we want split data by number of data is 1000
				# close tfrecord.
				writer.close()

				# make the new tfrecord.
				writer_idx+=1
				tfrecord_file = f'{tfrecord_path}/data{writer_idx}.tfrecords'
				writer = tf.io.TFRecordWriter(tfrecord_file)
			count+=1

	else :
		os.makedirs(f'{dca_path}', exist_ok=True)
		os.makedirs(f'{pssm_path}', exist_ok=True)
		for fasta in tqdm.tqdm(filtered_list):
			a3m = load_a3m(f'{a3m_path}/{fasta}.a3m')
			prpc_data = prpc_msa(a3m, msa_depth_max = msa_depth_max)
			np.save(f'{dca_path}/{fasta}', prpc_data['f2d_dca'])  # .npy
			np.savetxt(f'{pssm_path}/{fasta}.txt', prpc_data['f1d_pssm'])  # .txt


def parse_args(args) :
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers(dest="dataset_type", help='Arguments for specific dataset types. please choose numpy or tfrecord')
	subparsers.required = True

	parser.add_argument('--a3m_path', help="set a3m path")

	numpy_parser = subparsers.add_parser('numpy')
	numpy_parser.add_argument('--dca_path', help="set dca path")
	numpy_parser.add_argument('--pssm_path', help="set pssm path")

	tfrecord_parser = subparsers.add_parser('tfrecord')
	tfrecord_parser.add_argument('--tfrecord_path', help="set tfrecord path")
	tfrecord_parser.add_argument ('--dsmap_path', help="set distance path")

	parser.add_argument('--length_min', default=50, type=int,
						help='set your minimum value of sequence length, default value is 50')
	parser.add_argument('--length_max', default=300, type=int,
						help='set your maximum value of sequence length, default value is 300')
	parser.add_argument('--msa_depth_min', default=100, type=int,
						help='set your minimum number of msa result, default value is 100')
	parser.add_argument('--msa_depth_max', default=200, type=int,
						help='set your minimum number of msa result, default value is 200')

	parser.add_argument('--disable_gpu', action='store_true')
	parser.add_argument('--prpc_file', default=None,
						help='preprocessing pdb list file or none, none argument means that compute all data in a3m_path')

	return parser.parse_args(args)

def main(args=None):

	# todo parallel processing. -> no plan
	# todo check -> how many use msa result? (current 200)
	if args is None:
		args = sys.argv[1:]
	args = parse_args(args)

	if args.disable_gpu:
		tf.config.set_visible_devices([], 'GPU')
		logging.info('if you are using cpu, it takes a long time. you have to use gpu. tensorflow log is disable.')

	else:
		physical_devices = tf.config.list_physical_devices('GPU')
		tf.config.set_visible_devices(physical_devices[1], 'GPU')

	if args.dataset_type =='numpy' :
		save_data(a3m_path = args.a3m_path,
				  dca_path = args.dca_path,
				  pssm_path = args.pssm_path,
				  prpc_file = args.prpc_file,
				  length_min = args.length_min,
				  length_max = args.length_max,
				  msa_depth_min = args.msa_depth_min,
				  msa_depth_max = args.msa_depth_max,
				  mode='numpy')

	if args.dataset_type == 'tfrecord' :
		save_data(a3m_path = args.a3m_path,
				  dsmap_path=args.dsmap_path,
				  tfrecord_path = args.tfrecord_path,
				  prpc_file = args.prpc_file,
				  length_min=args.length_min,
				  length_max=args.length_max,
				  msa_depth_min=args.msa_depth_min,
				  msa_depth_max=args.msa_depth_max,
				  mode = 'tfrecord')


if __name__ == '__main__':
	main ()


