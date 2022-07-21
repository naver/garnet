"""
GaRNet
Copyright (c) 2022-present NAVER Corp.
Apache License v2.0
"""

#!/bin/bash

output_dir=./output
table3_dir=table.3
rm -rf $output_dir
mkdir -p $output_dir/$table3_dir

cd ./RELATED
model=$2

echo "========= evaluation on SCUT-EnsText dataset ============"
model=EnsNet
python ./eval.py --model_type EnsNet --model_path ../WEIGHTS/EnsNet/saved_model.pth --test_path ../DATA/JSON/$1/test.json --gpu | grep -E 'PSNR|GMac|Number of parameters' > ../$output_dir/$table3_dir/${model}.txt
echo "EnsNet"
cat ../$output_dir/$table3_dir/${model}.txt

model=MTRNet256
python ./eval.py --model_type MTRNet --model_path ../WEIGHTS/MTRNet/256/saved_model.pth --test_path ../DATA/JSON/$1/test.json --input_size 256 --gpu | grep -E 'PSNR|GMac|Number of parameters' > ../$output_dir/$table3_dir/${model}.txt
echo "MTRNet 256x256"
cat ../$output_dir/$table3_dir/${model}.txt

model=MTRNet512
python ./eval.py --model_type MTRNet --model_path ../WEIGHTS/MTRNet/512/saved_model.pth --test_path ../DATA/JSON/$1/test.json --gpu | grep -E 'PSNR|GMac|Number of parameters' > ../$output_dir/$table3_dir/${model}.txt
echo "MTRNet 512x512"
cat ../$output_dir/$table3_dir/${model}.txt

model=MTRNetPlusPlus256
python ./eval.py --model_type MTRNet++ --model_path ../WEIGHTS/MTRNet++/256/saved_model.pth --test_path ../DATA/JSON/$1/test.json --input_size 256 --gpu | grep -E 'PSNR|GMac|Number of parameters' > ../$output_dir/$table3_dir/${model}.txt
echo "MTRNet++ 256x256"
cat ../$output_dir/$table3_dir/${model}.txt

model=MTRNetPlusPlus
python ./eval.py --model_type MTRNet++ --model_path ../WEIGHTS/MTRNet++/512/saved_model.pth --test_path ../DATA/JSON/$1/test.json --gpu | grep -E 'PSNR|GMac|Number of parameters' > ../$output_dir/$table3_dir/${model}.txt
echo "MTRNet++ 512x512"
cat ../$output_dir/$table3_dir/${model}.txt

model=EraseNet
python ./eval.py --model_type EraseNet --model_path ../WEIGHTS/EraseNet/saved_model.pth --test_path ../DATA/JSON/$1/test.json --gpu | grep -E 'PSNR|GMac|Number of parameters' > ../$output_dir/$table3_dir/${model}.txt
echo "EraseNet"
cat ../$output_dir/$table3_dir/${model}.txt

cd ../CODE

model=GaRNet
python ./eval.py --model_path ../WEIGHTS/GaRNet/saved_model.pth --test_path ../DATA/JSON/$1/test.json --gpu | grep -E 'PSNR|GMac|Number of parameters' > ../$output_dir/$table3_dir/${model}.txt
echo "GaRNet"
cat ../$output_dir/$table3_dir/${model}.txt
