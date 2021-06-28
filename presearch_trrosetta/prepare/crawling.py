from bs4 import BeautifulSoup
from urllib import request
from urllib.error import HTTPError
import tqdm
import os
import argparse

def get_download(url, fname):
    try:
        request.urlretrieve(url, fname)
    except HTTPError as e:
        print('error')
        return

def get_casp_data(casps, download_path="./casp_data"):
    for casp in casps:
        os.makedirs(f'{download_path}', exist_ok=True)

        html = request.urlopen(f'https://predictioncenter.org/{casp}/targetlist.cgi')
        bsObject = BeautifulSoup(html, "html.parser")
        target_fasta = []

        for link in bsObject.find_all('a'):
            if "www.rcsb.org" in link.get('href'):
                # print(link.get(name))
                target_fasta.append(link.get_text())

        for name in tqdm.tqdm(set(target_fasta)):
            name = name.upper()
            download_link = f'https://files.rcsb.org/download/{name}.cif'
            get_download(download_link, f'{download_path}/{name}.cif')



def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_path', default='./cif')
    args = parser.parse_args()
    # todo : set the casp version.
    casps = ['casp12', 'casp13', 'casp14']
    get_casp_data(casps, args.download_path)


if __name__ == '__main__':
    main()