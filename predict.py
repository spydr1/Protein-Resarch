import sys
import numpy as np
import argparse
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tempfile
import shutil
import logging
import subprocess

import tensorflow as tf
import matplotlib.pyplot as plt

from presearch_trrosetta.architecture.trRosetta import trRosetta
from presearch_trrosetta.utils.config import DistanceConfig
from presearch_trrosetta.utils import vocab
from presearch_trrosetta.prepare.create_dataset import save_data


import pdb
# todo : tf code style
# todo : test code
# todo : argument collection
# todo : distance map
# todo : test for casp, parsing -> distance map
#
import tqdm

def predict_model(config):
    """
    get model
    """
    kwargs = dict(
        max_len=config.max_len,
        bins=config.bins,
        n2d_filters=64,
        n2d_layers=61,
        dropout_rate=0.15,
    )
    model = trRosetta(**kwargs)

    return model


def pred2dist(length,
              pred : tf.Variable):
    """
    convert the prediction to distance map.
    Return
        distance map : [width, height]
    """
    width, height , bins = pred.get_shape()
    mask = np.triu(np.ones([length,length]), k=1)
    argmax = tf.argmax(pred, axis=-1)[:length,:length]
    # argmax -> 0~16 , real dist -> 2~18
    argmax = ((argmax + 2) * mask).numpy()
    # distmap is symmetric.
    img = (argmax+argmax.T) .astype(np.float64)
    img *= 256 / bins
    return img


def pred2rr(length,
            pred : tf.Variable):
    """
    convert the prediction to rr format.
    Return
        rr : It is casp formant, [index i, index j, lower bound distance , upper bound distance, probability]
    """
    pred_numpy = pred.numpy()[ :length, :length]
    result_list = []
    for i in range(len(pred_numpy)):
        for j in range(len(pred_numpy)):
            # todo 1. : check i+4 or i+6
            if i + 1 < j:
                argmax = np.argmax(pred_numpy[i, j])
                prob = pred_numpy[i, j, argmax]
                if argmax < 6:
                    prob = pred_numpy[i, j, argmax]
                    result_list.append([i + 1, j + 1, argmax + 2, argmax + 3, prob])
                elif argmax < 10:
                    prob = pred_numpy[i, j, argmax]
                    result_list.append([i + 1, j + 1, 0, 12, prob])

    result_array = np.array(result_list)
    result_array = result_array[result_array[:, -1].argsort()[::-1]]

    return result_array



def get_prediction(inputs,
                   model : tf.keras.Model,
                   ):
    length = inputs['length'][0]
    prediction = model.predict_step(inputs)[0]

    distancemap = pred2dist(length, prediction)
    rr = pred2rr(length, prediction)

    return {"distancemap": distancemap,
            "rr": rr,}

def get_target_list(fasta_path,
                    a3m_path,
                    dca_path,
                    pssm_path):
    """
    check the it is existed that all of needed data. (f2d_dca, f1d_pssm)
    To make these two data, a3m file is must needed

    :return
        target_list : target fasta that prepared all of data.
        no_a3m_list : no a3m fasta (it is not applied msa - hhblits)
        no_dca_list : no dca fasta (it is not applied direct coupling analysis from baker)
    """
    target_list = []
    no_a3m_list = []
    no_dca_list = []
    for fasta_file in os.listdir(fasta_path):
        fasta_name,_ext = os.path.splitext(fasta_file)

        a3m_file = f"{a3m_path}/{fasta_name}.a3m"
        f2d_dca_file = f"{dca_path}/{fasta_name}.npy"
        f1d_pssm_file = f"{pssm_path}/{fasta_name}.txt"

        # for prediction, we need a3m, f2d_dca, f1d_pssm.
        if not os.path.exists(a3m_file):
            no_a3m_list.append(fasta_name)
        elif not (os.path.exists(f2d_dca_file) and os.path.exists(f1d_pssm_file)) :
            no_dca_list.append(fasta_name)
        else :
            target_list.append(fasta_name)

    return target_list, no_a3m_list, no_dca_list

def get_seq(fasta_file,
            max_len):
    """
    For the matching the maximum length, cutting or padding the seq
    :param fasta_file : .seq file
    :param max_len : maximum length of fasta.

    :return:
        seq : one-hot & padded or cut sequence array
        length : length of fasta

        ex)
        seq : [[1, 0, 0 .. ], [0, 1, 0 ...], ...]
        length : 50
    """
    with open(fasta_file, "r") as output:
        fasta_name = output.readline().rstrip()
        seq = output.readline().rstrip()

    assert type(seq) != 'str', 'check the seq_file, it must have fasta name line and AA seq line'
    #assert '\n' in seq, 'please exclude \n'

    seq = seq[:max_len]
    length = len(seq)
    if len(seq)<max_len:
        seq += '-' * (max_len - len(seq))
    seq = [vocab.onehot_dict[_seq.upper()] for _seq in seq]
    return np.array(seq), length

def get_pssm(pssm_file,
             max_len):
    """
    For the matching the maximum length, cutting or padding the pssm
    :param pssm_file:
    :param max_len : maximum length of fasta.

    :return : padded or cut pssm array
    """

    pssm = np.loadtxt(pssm_file)
    pssm = pssm[:max_len]
    if  len(pssm) < max_len:
        # padding
        pssm = np.vstack([pssm, np.zeros([max_len - pssm.shape[0], 21])])
    return pssm


def get_dca(msa_file,
            max_len):
    """
    For the matching the maximum length, cutting or padding the dca.

    :param msa_file: .npy file
    :param max_len : maximum length of fasta.

    :return : padded or cut dca array - [max_len,max_len,441]
    """
    msa_prpc = np.load(msa_file)
    msa_prpc = msa_prpc[:max_len, :max_len, :]
    pad_len = max(0, max_len - msa_prpc.shape[0])
    # todo : log ?

    if msa_prpc.shape[0] < max_len:
        msa_prpc = np.pad(msa_prpc, [[0, pad_len], [0, pad_len], [0, 0]])

    return msa_prpc

def save_rr(output_path,
            fasta,
            rr,
            seq
            ):
    """
    save the rrfile
    """
    with open(f'{output_path}/{fasta}/{fasta}.rr', mode='w') as file_obj:
        file_obj.write(''.join([vocab.onehot_dict_inv[s] for s in seq]) + '\n')
        for idx, _array in enumerate(rr):
            if idx + 1 == len(rr):
                file_obj.write('{:.0f} {:.0f} {:.0f} {:.0f} {:f}'.format(*_array))
            else:
                file_obj.write('{:.0f} {:.0f} {:.0f} {:.0f} {:f}\n'.format(*_array))

def save_distancemap(output_path,
                     fasta,
                     distancemap,
                     ):
    plt.imsave(f'{output_path}/{fasta}/{fasta}.png',distancemap)

def check_train_data(target_fasta,
                     train_fasta_list):
    """
    check the is it trained fasta ?
    """

    for fasta in train_fasta_list:
        if target_fasta[:4] in fasta:
            return True
            break

    return False

def makedirs(*path):
    for _path in path :
        os.makedirs(_path,exist_ok=True)

def prpc_data(fasta_path,
              a3m_path,
              dca_path,
              pssm_path,
              database_path,
              length_min = 50,
              length_max = 300,
              msa_depth_min = 100,
              msa_depth_max = 200,):

    target_fasta_list, no_a3m_list, no_dca_list = get_target_list(fasta_path, a3m_path, dca_path, pssm_path)
    print(target_fasta_list, no_a3m_list, no_dca_list)
    logging.info(f"need the pre-processing. we have to create a3m, f1d_pssm, f2d_dca. "
                 f"number of data that needed processing. : {len(no_a3m_list) + len(no_dca_list)}")

    with tempfile.TemporaryDirectory() as tmp_path:
        tmp_fasta_path = f'{tmp_path}/fasta'
        tmp_a3m_path = f'{tmp_path}/a3m'
        makedirs(tmp_fasta_path, tmp_a3m_path)

        # absPath =os.path.abspath(__file__)
        # curPath, file =  os.path.split(absPath)

        # copy the fasta from data_path to tmp_path
        for no_a3m_data in no_a3m_list:
            logging.info(no_a3m_data)
            src_fasta = f'{fasta_path}/{no_a3m_data}.fasta'
            dst_fasta = f'{tmp_fasta_path}/{no_a3m_data}.fasta'
            shutil.copy(src_fasta, dst_fasta)

        # subprocess.call(["python ../msa_hhblits_mp.py --fasta_dir={tmp_path}/fasta --a3m_dir={tmp_path}/a3m --database_dir=/uniclust/uniclust30_2018_08 --cpu=8"])
        subprocess.call(["python", "./presearch_trrosetta/prepare/create_a3m.py",
                         "--fasta_path", tmp_fasta_path,
                         "--a3m_path", tmp_a3m_path,
                         "--database_path", database_path,
                         "--cpu", "8"])
        # todo : seq -> fasta

        # move the finished a3m file from tmp_path to output_path
        for finished_a3m_data in no_a3m_list:
            src_a3m = f'{tmp_a3m_path}/{finished_a3m_data}.a3m'
            dst_a3m = f'{a3m_path}/{finished_a3m_data}.a3m'
            shutil.copy(src_a3m, dst_a3m)

        print("a3m finish")
        # copy the a3m from data_path to tmp_path
        for no_dca_data in no_dca_list:
            src_a3m = f'{a3m_path}/{no_dca_data}.a3m'
            dst_a3m = f'{tmp_a3m_path}/{no_dca_data}.a3m'
            shutil.copy(src_a3m, dst_a3m)

        save_data(a3m_path=tmp_a3m_path,
                  dca_path=dca_path,
                  pssm_path=pssm_path,
                  length_min=length_min,
                  length_max=length_max,
                  msa_depth_min=msa_depth_min,
                  msa_depth_max=msa_depth_max,
                  mode='numpy')

        logging.info("pre-proceesing is finished.")

def predict(fasta_path,
            a3m_path,
            dca_path,
            pssm_path,
            output_path,
            config,
            experiment_folder,
            ):

    model = predict_model(config)
    model.load_weights(config.load_weight)
    #model.summary()

    # we want to check that target is already trained or not.
    target_fasta_list, no_a3m_list, no_dca_list = get_target_list(fasta_path, a3m_path, dca_path, pssm_path)
    trained_fasta_list = np.loadtxt(f'{experiment_folder}/train_list.txt', dtype=str)

    for fasta in tqdm.tqdm(target_fasta_list, desc="predict"):

        # todo : using the __getitem__ or not.
        seq, length = get_seq(f'{fasta_path}/{fasta}.fasta', config.max_len)
        f1d_pssm = get_pssm(f'{pssm_path}/{fasta}.txt', config.max_len)
        f2d_dca = get_dca(f'{dca_path}/{fasta}.npy', config.max_len)

        inputs = {'seq': np.expand_dims(seq,axis=0),
                  'f2d_dca': np.expand_dims(f2d_dca,axis=0),
                  'f1d_pssm': np.expand_dims(f1d_pssm,axis=0),
                  'length' : [length]}

        prediction = get_prediction(inputs,
                                    model
                                    )

        os.makedirs(f'{output_path}/{fasta}/', exist_ok=True)
        save_rr(output_path,fasta,prediction['rr'],seq[:length])
        save_distancemap(output_path,fasta,prediction['distancemap'][:length,:length])

        with open(f'{output_path}/{fasta}/logs.txt', mode='w') as file_obj:
            if check_train_data(fasta,trained_fasta_list):
                file_obj.write("trained")
            else:
                file_obj.write("not-trained")

        print(f"prediction is finished, check your output fordler{output_path}")


def parse_args(args) :
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment_folder', default=None)
    parser.add_argument('--fasta_path', help="set fasta path")
    parser.add_argument('--a3m_path', help="set a3m path")
    parser.add_argument('--dca_path', help="set dca path")
    parser.add_argument('--pssm_path', help="set pssm path")
    parser.add_argument('--output_path', help="set your output path")
    parser.add_argument('--database_path', help="set your Uniclust database")

    parser.add_argument('--length_min', default=50, type=int,
                        help='set your minimum value of sequence length, default value is 50')
    parser.add_argument('--length_max', default=300, type=int,
                        help='set your maximum value of sequence length, default value is 300')

    parser.add_argument('--msa_depth_min', default=100, type=int,
                        help='set your minimum number of msa result, default value is 100')
    parser.add_argument('--msa_depth_max', default=200, type=int,
                        help='set your minimum number of msa result, default value is 200')

    return parser.parse_args(args)


def main(args=None):

	# todo parallel processing. -> no plan
	# todo check -> how many use msa result? (current 200)
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    if args.experiment_folder == None :
        config = DistanceConfig()
    else :
        if os.path.isfile(f'{args.experiment_folder}/config.json'):
            print("config file exist")
            config = DistanceConfig.from_json_file(f'{args.experiment_folder}/config.json')
        else :
            config = DistanceConfig()
            with open(f'{args.experiment_folder}/config.json', 'w', encoding='utf-8') as config_file :
                json.dump(config.to_dict(), config_file)

    config.load_weight = f'{args.experiment_folder}/{config.load_weight}'

    makedirs(args.fasta_path, args.a3m_path, args.dca_path, args.pssm_path)

    prpc_data(
        fasta_path = args.fasta_path,
        a3m_path = args.a3m_path,
        dca_path = args.dca_path,
        pssm_path = args.pssm_path,
        database_path = args.database_path,
        length_min = args.length_min,
        length_max = args.length_max,
        msa_depth_min = args.msa_depth_min,
        msa_depth_max = args.msa_depth_max,
    )

    predict(
        fasta_path = args.fasta_path,
        a3m_path = args.a3m_path,
        dca_path = args.dca_path,
        pssm_path = args.pssm_path,
        output_path = args.output_path,
        config = config,
        experiment_folder = args.experiment_folder,
    )


if __name__ == '__main__':
    main()

# todo : DCon까지
# evaluate
# report 다른것도 추가하고
# cif 전처리 까지 추가 해서 create data
# nmr 구조는 여러개의 체인이 중복 (실험을 여러번 해서 ? )
# .seq -> .fasta