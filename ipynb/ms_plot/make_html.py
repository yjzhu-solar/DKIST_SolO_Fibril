from glob import glob 
from ipy2html import convert 
from fix_links import process_file
import os 
from pathlib import Path

ipynb_files = glob("*.ipynb")
ipynb_files += ["/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/ipynb/trace_fibril/bg_removal_test.ipynb"]

for ipynb_file in ipynb_files:
    convert(ipynb_file, output_dir="/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/ipynb_html/")

html_files = glob("/cluster/home/zhuyin/Solar/DKIST_SolO_Fibril/ipynb_html/*.html")

for html_file in html_files:
    process_file(Path(html_file), "https://yjzhu-solar.github.io/DKIST_SolO_Fibril", "ipynb_html")