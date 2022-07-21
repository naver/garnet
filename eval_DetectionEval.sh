"""
GaRNet
Copyright (c) 2022-present NAVER Corp.
Apache License v2.0
"""

#!/bin/bash

output_dir=./output
table3_dir=detector.3
rm -rf $output_dir
rm -rf ./RELATED/result
rm -rf ./CRAFT/result

mkdir ./CRAFT/result
mkdir -p $output_dir/$table3_dir

cd ./RELATED/
model=$2
if [ "${model}" == 'MTRNet' ]||[ "${model}" == 'MTRNet++' ]
then
    model_PATH="../WEIGHTS/${model}/512/saved_model.pth"
else
    model_PATH="../WEIGHTS/${model}/saved_model.pth"
fi

echo "========= DetEVAL ============"
echo "${model}"
python ./test.py --result_path ./result --test_path ../DATA/JSON/$1/test.json --use_json --model_type ${model} --model_path $model_PATH  --input_size 512 --gpu --use_composited > ../$output_dir/$table3_dir/${model}.txt
cd ../CRAFT
python ./test.py --trained_model=./craft_mlt_25k.pth --test_folder=../RELATED/result
cd ../
python ./make_detection_zip.py --path ./CRAFT/result --save_path ./CRAFT/result/tmp
cd ./CRAFT/result/tmp
zip ../submit.zip * > ../../../$output_dir/$table3_dir/${model}.txt
cd ../../../DETEVAL
python script.py -g=../DATA/DetectionGT/$1/gt.zip -s=../CRAFT/result/submit.zip > ../$output_dir/$table3_dir/${model}.txt
cd ../RELATED
cat ../$output_dir/$table3_dir/${model}.txt
