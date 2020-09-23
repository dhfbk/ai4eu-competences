import argparse
import re
import os
import urllib.request
from tqdm import tqdm
from warcio.archiveiterator import ArchiveIterator
import gzip

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


parser = argparse.ArgumentParser(description='Process WET files from Common Crawl')
parser.add_argument('inputFile', help='Input file with paths')
parser.add_argument('outputFolder', help='Output folder')
args = parser.parse_args()

inputFile = args.inputFile
outputFolder = args.outputFolder

filePattern = re.compile(".*segments/([0-9.]+)/wet/([^.]*)(.*)")
file_prefix = 'https://commoncrawl.s3.amazonaws.com/'

if not os.path.exists(outputFolder):
    os.mkdir(outputFolder)

with open(inputFile, "r") as f:
    for line in f:
        line = line.strip()
        res = filePattern.match(line)
        if res:
            folder = os.path.join(outputFolder, res.group(1))
            if not os.path.exists(folder):
                os.mkdir(folder)
            outputFile = os.path.join(folder, res.group(2) + ".gz")
            print("File:", res.group(2))
            if os.path.exists(outputFile):
                print("File exists, skipping")
                continue
            tmpFile = os.path.join(outputFolder, "tmp")
            url = file_prefix + line

            download_url(url, tmpFile)

            with gzip.open(outputFile, "w") as fw:
                with open(tmpFile, 'rb') as stream:
                    for record in ArchiveIterator(stream):
                        langs = record.rec_headers.get_header("WARC-Identified-Content-Language")
                        if langs is not None and langs.startswith("ita"):
                            text = record.raw_stream.read()
                            fw.write(text)
                            fw.write(b"\n")

            os.remove(tmpFile)


