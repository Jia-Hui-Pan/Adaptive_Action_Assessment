<h1 align = "center"> Adaptive Action Assessment </center></h1>
<h4 align = "center"><a href="https://scholar.google.com/citations?user=c0UQD6oAAAAJ&hl=zh-CN">Jia-Hui Pan</a>, Jibin Gao, and Wei-Shi Zheng</h4>
<h4 align = "center">Sun Yat-sen University.</h4>

### Introduction
This repository is for our TPAMI 2021 paper "[Adaptive Action Assessment](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9609694)" which proposes to adaptively design different assessment architectures for different types of actions.

<div style="text-align: center;">
    <img style="border-radius: 0.5125em;
    width: 88%;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src=./teaser_adaptive_action_assessment.jpg alt="">
    <br>
</div>

### Network Architecture
The overall structure of the proposed network is shown below. Our model takes the whole-scene and local-patch videos (with the local patches cropped around joints) as
input and extracts video features by I3D Network. Then our model performs interactive joint motion pattern modelling with the local-patch features by
learning body part kinetics and joint coordination on trainable joint relation graphs. The interactive joint motion patterns and the whole-scene features
are concatenated to form a motion tensor. After that, our model learns a specific assessment function architecture for each type of action which consumes the motion tensor to learn the assessment results.

<div style="text-align: center;">
    <img style="border-radius: 0.3125em;
    width: 98%;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src=./framework_adaptive_action_assessment.jpg alt="">
    <br>
</div>


### Installation
 Use Conda to install the environment. For example,
```
conda env create -f environment.yml
```

 And then activate the environment using
```
conda activate assessment
```


 Please follow the instructions for testing the model's performance on the AQA-7 dataset.
 1. Download and extract the [features](https://drive.google.com/file/d/1N8ZT9yxT9p7T1A9zjVogtXsC35QHCvWQ/view?usp=sharing) to the "./feature" directory.
 2. Download and extract the [checkpoints](https://drive.google.com/file/d/1rfFzLiM0imm8zcvrHVdeLWO2_7twnzjw/view?usp=sharing) to the "./checkpoints" directory.
 3. Run the evaluation code. 
```
python evaluation.py -gpu=0 -mode=structure -set-id=0 -loss-type=new_mse

# The '-set-id' can be set as 0, 1, 2, 3, 4 or 5, 
# each representing one type of action in the AQA-7 dataset.
```
