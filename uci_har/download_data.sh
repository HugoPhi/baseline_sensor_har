#!/bin/bash

wget https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI\ HAR\ Dataset.zip
unzip UCI\ HAR\ Dataset.zip
rm -rf ./__MACOSX
mv ./UCI\ HAR\ Dataset data
rm UCI\ HAR\ Dataset.zip 
