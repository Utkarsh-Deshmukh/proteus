#!/bin/bash
clear
apt install python3.10-venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
apt install pre-commit
pre-commit install
