# The Surprisingly Straightforward Scene Text Removal Method With Gated Attention and Region of Interest Generation: A Comprehensive Prominent Model Analysis

> The Surprisingly Straightforward Scene Text Removal Method With Gated Attention and Region of Interest Generation: A Comprehensive Prominent Model Analysis | [Paper and Supplementary](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4705_ECCV_2022_paper.php) \
> Hyeonsu Lee, Chankyu Choi \
> Naver Corp. \
> In ECCV 2022.

This repository is official Pytorch implementation of GaRNet

### Sample results
The sample image was taken from SCUT-EnsText.
<img width="100%" src=ASSETS/sample1.png>
<img width="100%" src=ASSETS/sample2.gif>


# Getting started

### Requirements
* Python 3.6.9
* pytorch : 1.2.0
* numpy 1.16.5
* opencv 4.5.1.48
* torchvision==0.4.0
* ptflops==0.6.4 (for calculate inference time (GPU Time), GMac)
* scikit-image==0.17.2 (for CRAFT)

The requirements can be installed by running
      
      pip install -r requirements.txt

### Dataset

To compare the performance of previously proposed methods with the same evaluation dataset. we combined Oxford-synthetic data and SCUT-EnsText data.
We provide subset information of combined dataset.

* Full data of SCUT-EnsText can be downloaded at SCUT-EnsText github : https://github.com/HCIILAB/SCUT-EnsText
* Full data of Oxford-synthetic data and coressponding background images can be downloaded at 

  : Synthext: https://www.robots.ox.ac.uk/~vgg/data/scenetext/ 
  
  : bg_img: https://github.com/ankush-me/SynthText

* Subset information of Oxford Synthetic dataset is in .json file which exisit in [link](https://drive.google.com/drive/folders/1CewJe9BYfgpOYC5QkfsVwiXZC8ws_n77?usp=share_link)
  - you have to move that .json file to DATA/JSON/SYN
* The json file of SCUT-EnsText is in DATA/JSON/REAL
* For evaluation, SynthText must be placed in ./DATA/SYN and SCUT-EnsText in ./DATA/REAL

### Pre-trained models

We provide the weight of EnsNet, MTRNet, MTRNet++, EraseNet and Our's trained on our combined dataset.

* You can find pre-trained models in [drive](https://drive.google.com/drive/folders/18bLTPXHSfjjITCDbp9599S3yMwjg6imX?usp=share_link)

Note that all models are pre-trained by Synthetic datasets and SCUT-EnsText, and the SCUT-EnsText dataset can only be used for non-commercial research purposes.

### Instructions for Comparison between Related works (Table 2 and 3 in the paper)

ImageEval

* All

            bash eval_ImageEval_All.sh [REAL or SYN]

* Individual

            cd ./RELATED
            python ./eval.py --model_type=[model type (ex. EnsNet, GaRNet...)] default: EnsNet \\
                  --model_path=[pretrained model path] default: ../WEIGHTS/EnsNet/saved_model.pth \\
                  --test_path=[the path of Test json file] default: ../DATA/JSON/REAL/test.json \\
                  --input_size=[size of input image] default: 512 \\
                  --batch=[batch size] default: 10 \\
                  --gpu (use gpu for evaluation) default: False

For MTRNet++ and EraseNet, you need to get official model (or network) codes of them.

* MTRNet++: https://github.com/neouyghur/One-stage-Mask-based-Scene-Text-Eraser/blob/main/src/networks.py

  copy src/networks.py to RELATED/MTRNet++/src/networks.py
  
* EraseNet: https://github.com/lcy0604/EraseNet/blob/master/models

  copy models/sa_gan.py to RELATED/EraseNet/models/sa_gan.py
  
  copy models/networks.py to RELATED/EraseNet/models/networks.py

  Note that for Erasenet, mask branches should be erased from sa_gan.py when measuring size of models params or speed. (EraseNet does not use mask branches when inference)

DetectionEval (GPU is required to run CRAFT)
      
      bash eval_DetectionEval.sh [REAL or SYN] [Model type (ex. EnsNet, GaRNet ...)]
      
For Detection Eval, you need to get CRAFT and DetEVAL script.

* CRAFT: https://github.com/clovaai/CRAFT-pytorch (use General model)
  
  Download code and pretrained model to ./CRAFT
  
* DetEVAL: https://rrc.cvc.uab.es/?ch=1&com=mymethods&task=1 (need registration)

  Download script file to ./DETEVAL

### Instructions for proposed method

For Eval

      cd ./CODE
      python ./eval.py --model_path=[pretrained model path] default: ../WEIGHTS/GaRNet/saved_model.pth \\
            --test_path=[the path of Test json file] default: ../DATA/JSON/REAL/test.json \\
            --batch=[batch size] default: 10 \\
            --gpu (use gpu for evaluation) defulat: False \\

For Inference
      
      cd ./CODE
      python ./inference.py --result_path=[the path for save output image] default: ./result \\
            --image_path=[the path of input image] default: ../DATA/IMG \\
            --box_path=[the path of text files which have box information] default: ../DATA/TXT \\
            --input_size=[inference size] default: 512 \\
            --model_path=[the path of trained model] default: ../WEIGHTS/GaRNet/saved_model.pth \\
            --attention_vis (visualize attention) default: False \\
            --gpu (use gpu for inference) default: False \\

# Citation
Please cite our paper if this work is useful for your research.

```
@inproceedings{lee2022surprisingly,
  title={The Surprisingly Straightforward Scene Text Removal Method with Gated Attention and Region of Interest Generation: A Comprehensive Prominent Model Analysis},
  author={Lee, Hyeonsu and Choi, Chankyu},
  booktitle={European Conference on Computer Vision},
  pages={457--472},
  year={2022},
  organization={Springer}
}
```

# License
GaRNet is licensed under Apache-2.0. See [LICENSE](/LICENSE) for the full license text.
```
Copyright 2022-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

# Acknowledgments

Our model codes starts from [EnsNet](https://github.com/HCIILAB/Scene-Text-Removal). \
Our research is benefit a lot from [MTRNet++](https://github.com/neouyghur/One-stage-Mask-based-Scene-Text-Eraser/blob/main/src/networks.py) and [EraseNet](https://github.com/lcy0604/EraseNet/blob/master/models), we wish to thank their work for providing model codes and real-dataset. \
We acknowledge the official code and pre-trained weight [CRAFT](https://github.com/clovaai/CRAFT-pytorch) \
We use ssim calculation code in [piq](https://github.com/photosynthesis-team/piq).
