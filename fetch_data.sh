#!/usr/bin/env bash
# Downloads the Wasa motorcycle insurance data (Ohlsson & Johansson, 2010)
# from the Rdatasets collection of the R package "insuranceData".
set -euo pipefail
mkdir -p data
curl -fsSL "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/insuranceData/dataOhlsson.csv" -o data/ohlsson.csv
echo "Saved data/ohlsson.csv"
