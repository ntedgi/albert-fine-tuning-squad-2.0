#!/bin/bash
set -e
set -x

virtualenv -p python3.6 albert
albert/bin/pip3.6 install -r requirements.txt
