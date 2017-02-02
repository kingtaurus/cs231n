import gzip
import sys
import os

from xml.etree import ElementTree

import urllib.request as request
from urllib.request import urlopen
from urllib.error import  URLError, HTTPError

import requests

class Config():
    def __init__(self):
        self.synset_url = "http://www.image-net.org/download/synset"
        self.username = ""
        self.accesskey = ""
        self.filepath = "images"

        self.base_url = "http://www.image-net.org/api/xml/"
        self.structure_released = "structure_released.xml"

def get_imagepath(wnid):
    return os.path.join(config.filepath, wnid + ".tar")

# inF = gzip.open(file, 'rb')
# outF = open(outfilename, 'wb')
# outF.write( inF.read() )
# inF.close()
# outF.close()
##reading a gzipped file


# >>> req = urllib.request.Request('http://www.pretend_server.org')
# >>> try: urllib.request.urlopen(req)
# >>> except urllib.error.URLError as e:
# >>>    print(e.reason)

# req = Request(someurl)
# try:
#     response = urlopen(req)
# except URLError as e:
#     if hasattr(e, 'reason'):
#         print('We failed to reach a server.')
#         print('Reason: ', e.reason)
#     elif hasattr(e, 'code'):
#         print('The server couldn\'t fulfill the request.')
#         print('Error code: ', e.code)
# else:

#wget http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz
#wget http://www.image-net.org/api/xml/structure_released.xml
def download_file(url, dst, params={}, debug=True):
    if debug:
        print("downloading {0} ({1})...".format(dst, url))
    params.get('User-Agent', "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36")
    response = requests.get(url, params=params)
    content_type = response.headers["content-type"]
    if content_type.startswith("text"):
        raise TypeError("404 Error")
    with open(dst, "wb") as out_file:
        out_file.write(response.content)
    print("done.")

def download_text(url, dst, params={}, debug=True):
    if debug:
        print("downloading {0} ({1})...".format(dst, url))
    params.get('User-Agent', "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36")
    response = requests.get(url, stream=True, params=params)
    content_type = response.headers["content-type"]
    if content_type.startswith('text'):
        with open(dst, 'wb') as out_file:
            out_file.write(response.content)


def main():
    config = Config()
    print("Starting to download image-net images")
    if not os.path.exists(config.structure_released):
        print("The file {0} does not exists.".format(config.structure_released))
        download_text(config.base_url + config.structure_released, config.structure_released)
        # download_file(config.base_url + config.structure_released,
        #               config.structure_released)
    with open('fall11_urls.txt', 'r') as file:
        for idx, line in enumerate(file):
            if idx > 10:
                break
            name, url = line.split()
            try:
                request.Request(url)
            except HTTPError as e:
                print('The server couldn\'t fulfill the request.')
                print('Error code: ', e.code)
            except URLError as e:
                print('We failed to reach a server.')
                print('Reason: ', e.reason)
            else:
                print("fine: " + name)
                response = requests.get(url, stream=True, headers={'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'})
                content_type = response.headers["content-type"]
                if content_type.startswith("text"):
                    print(response, name)
                    print(url)
                    #print(response.content)
                    continue

            # with open(name + '.jpg', 'wb') as image:
            #     response = requests.get(url, stream=True, headers={'User-Agent':'test'})
            #     content_type = response.headers["content-type"]
            #     if content_type.startswith("text"):
            #         print(response, name)
            #         continue
            #     image.write(response.content)

if __name__ == '__main__':
    main()
