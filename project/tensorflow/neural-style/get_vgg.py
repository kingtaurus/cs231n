#!/usr/bin/env python3

import os
import sys
import hashlib

import requests

# use 16 byte chunks
BUF_SIZE = (2 << 16)

TEST_MD5_HASH    = "483ac245fd363b342b99c4717775744d"
TEST_SHA256_HASH = "da8951134a14d06287616abe84749d04960d5c2a4e8493dab018f87d8eae8d12"

VGG_MD5_HASH = "106118b7cf60435e6d8e04f6a6dc3657"
VGG_SHA256_HASH = "abdb57167f82a2a1fbab1e1c16ad9373411883f262a1a37ee5db2e6fb0044695"


DATA_DIR = "vgg_weights"

def block_iter(in_file, block_size=BUF_SIZE):
    with open(in_file, 'rb') as file:
        block = file.read(block_size)
        while len(block) > 0:
            yield block
            block = file.read(block_size)
    return StopIteration

def hash_file(in_file):
    sha256 = hashlib.sha256()
    for chunk in block_iter(in_file):
        sha256.update(chunk)
    return sha256.hexdigest()

def md5_hash_file(in_file):
    md5 = hashlib.md5()
    with open(in_file, 'rb') as file:
        for chunk in iter(lambda: file.read(BUF_SIZE), b''):
            #iterate till file.read(BUF_SIZE) returns b''
            md5.update(chunk)
    return md5.hexdigest()

def maybe_download(link, dest_directory=DATA_DIR, expected_bytes=0):
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    file_name = link.split('/')[-1]
    file_path = os.path.join(dest_directory, file_name)
    if not os.path.exists(file_path):
        r = requests.get(link, stream=True)
        if r.status_code == requests.codes.ok:
            total_length = int(r.headers.get('content-length'))
            with open(file_path, 'wb') as file:
                dl = 0
                for chunk in r.iter_content(chunk_size=BUF_SIZE):
                    dl += len(chunk)
                    file.write(chunk)
                    done = int(50 * dl / total_length)#[every 2%]
                    sys.stdout.write("\r[%s %.1f%%] [%s%s]" % (file_name, 100. * dl / total_length, '=' * done, ' ' * (50-done)) )    
                    sys.stdout.flush()
    print()
    statinfo = os.stat(file_path)
    print('Successfully downloaded', file_name, statinfo.st_size, 'bytes.')

def download(link, file_name, expected_bytes=0):
    # if os.path.exists(file_name):
    #     if hash_file(file_name) !=
    if os.path.exists(file_name):
        if TEST_MD5_HASH == md5_hash_file(file_name):
            print("Already have file")
            return file_name
    r = requests.get(link)
    if r.status_code == requests.codes.ok:
        with open(file_name, 'wb') as file:
            for chunk in r.iter_content(chunk_size=BUF_SIZE):
                file.write(chunk)
    else:
        r.raise_status_code() 
    if TEST_SHA256_HASH != hash_file(file_name):
        raise Exception('file ' + file_name + " doesn't have matching hash; ")
    return file_name

if __name__ == '__main__':
    print(md5_hash_file("get_vgg.py"))
    print(hash_file("get_vgg.py"))
    download("http://docs.python-requests.org/en/master/_static/requests-sidebar.png", "request-icon.png")
    maybe_download(link='http://docs.python-requests.org/en/master/_static/requests-sidebar.png')
    maybe_download(link='http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat')
    print(md5_hash_file('vgg_weights/imagenet-vgg-verydeep-19.mat'))
    print(hash_file('vgg_weights/imagenet-vgg-verydeep-19.mat'))
