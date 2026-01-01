# Adhesion-and-polarity-driven-morphogenesis
====

Overview

## Description
Source code of the paper. 

## Requirement
C++ code: Eigen 3.3.8 / compiler: g++ 8.2.0

python code: pandas numpy matplotlib opencv-python networkx ripser 
## Usage
```
cd src

mkdir -p ../data_model_base_2D
mkdir -p ../data_model_base_3D
mkdir -p ../fig
mkdir -p ../feature_analysis
```
### 2D
```
g++ -O3 -g model_base_2D.cpp -o model_base_2D
./job_model_base_2D.sh
```
### 3D
```
g++ -O3 -g model_base_3D.cpp -o model_base_3D
./job_model_base_3D.sh
```
## Licence

[MIT](https://github.com/Nakayoshi98/Adhesion-and-polarity-driven-morphogenesis/blob/main/LICENSE)

## Reference
