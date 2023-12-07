#!bin/bash
source /home/smit/Dev/flask/flask_file_upload/venv/bin/activate
pip install -q git+https://github.com/huggingface/transformers.git
pip install -q datasets seqeval
pip install transformers
pip install torch