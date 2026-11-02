#!/bin/bash
cd "$(dirname "$0")"
until grep -q ALLDONE runall.log; do sleep 60; done
python3 -W ignore fit.py > fit_all.txt 2>&1
echo FITDONE >> fit_all.txt
