import logging
import os
import sys

import numpy as np
import argparse

from PIL import Image
from Bio.PDB import MMCIFParser
from scipy.spatial import distance_matrix

import pdb
import tqdm
from presearch_trrosetta.utils.vocab import aa_dict

def save_fasta(res_name, fasta_path, seq):
    with open(os.path.join(fasta_path, f'{res_name}.fasta'), mode='w') as obj:
        obj.write(f'>{res_name}\n')
        obj.write(seq)

def create_dsmap(cif_path,
                 dsmap_path,
                 fasta_path):
    p = MMCIFParser()
    for name in tqdm.tqdm(os.listdir(cif_path)):
        structure = p.get_structure(name, f"{cif_path}/{name}")
        for model in structure:
            for chain in model:
                pdb_id = os.path.splitext(name)[0]
                res_name = f'{pdb_id.upper()}_{chain.id}'

                coords = []
                seqs = []

                # todo : how to treat the hetaatom, UNK case, no ss case
                # missing part is not in structure line. It is written another line.
                # anyway in biopython module, not read missing part. It is skipped
                # todo : in NMR,because of many experiment to identical residue, there is many redundant chain. So it is needed more time.
                for amino in chain:
                    if amino.get_id()[0] == ' ':
                        coords.append(amino["CA"].get_coord())
                        if amino.get_resname()!='UNK':
                            seqs.append(aa_dict[amino.get_resname()])

                logging.info(f"{res_name} - num of coords : {len(coords)}")
                if len(coords)>0:
                    # save img
                    try:
                        coords = np.array(coords)
                        gt_distance_matrix = distance_matrix(coords, coords)
                        im = Image.fromarray(gt_distance_matrix.astype(np.int8))
                        im.save(os.path.join(dsmap_path, f'{res_name}.png'))

                    except :
                        #pdb.set_trace()
                        logging.warning(f"check the {res_name}")
                    # save seq
                    save_fasta(res_name, fasta_path, ''.join(seqs))

def parse_args(args) :
    parser = argparse.ArgumentParser()
    parser.add_argument('--cif_path')
    parser.add_argument('--dsmap_path')
    parser.add_argument('--fasta_path')

    return parser.parse_args(args)

def make_dirs(*path):
    for _path in path :
        os.makedirs(_path, exist_ok=True)

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    make_dirs(args.dsmap_path,args.fasta_path)

    create_dsmap(args.cif_path,
                 args.dsmap_path,
                 args.fasta_path)

if __name__ == '__main__' :
    main()
# todo : multiprocessing ?
