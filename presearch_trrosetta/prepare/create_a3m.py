# -*- coding: utf-8 -*-
__author__ = "Yechan Hong"
__maintainer__ = "Yechan Hong"
__email__ = "ychuh@pharmcadd.com"
__status__ = "Dev"


import argparse
import multiprocessing
import os
import time
import subprocess



def a3m__seq(seq_fp, a3m_fp, database, e=.001, n=1, cpu=1):
    ''' A single function encapsulating hhblits
    seq_fp: path to seq file
    a3m_fp: path to save output
    e: E-value
    n: number of ierations
    cpu: cpu to use. It has been noted online and verified experimentally that running separate processes is faster than increasing this cpu value
    '''
    os.system('hhblits -o temp.hhr -cpu '+str(cpu)+' -i '+seq_fp+' -d '+database+' -oa3m '+a3m_fp+' -n '+str(n)+ ' -e '+str(e))

if __name__ == '__main__' : 
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta_path", type=str, help="sequence information load directory", required=True)
    parser.add_argument("--a3m_path", type=str, help="a3m alignment save directory", required=True)
    parser.add_argument("--database_path", type=str, help="database to hhblits", required=True)
    parser.add_argument("--cpu", type=int, default=None, help="number of cpu to use")
    args = parser.parse_args()

    database = args.database_path
    load_path = args.fasta_path
    save_path = args.a3m_path
    os.makedirs(save_path,exist_ok=True)

    N_WORKERS = args.cpu #None #This will use all available cpu

    # Build TASKS of hhblits
    TASKS = []
    S = [ os.path.splitext(s)[0] for s in os.listdir(load_path) ]
    for s in S:
        in_fp = os.path.join(load_path, s+'.fasta')
        out_fp = os.path.join(save_path, s+'.a3m')
        TASKS.append( (in_fp, out_fp, database) )

    # Run mp hhblits pool
    try:
        pool = multiprocessing.Pool(N_WORKERS)
        r = pool.starmap(a3m__seq, TASKS)
    except:
        pool.close()
        pool.terminate()
    pool.terminate()
