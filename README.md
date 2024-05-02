<h2 align = "center"> Adaptive_Action_Assessment </center></h2>
<h4 align = "center">Jia-Hui Pan<sup>1</sup>, Jibin Gao<sup>1</sup>, and Wei-Shi Zheng<sup>1</sup></h4>
<h4 align = "center"> <sup>1</sup>Sun Yat-sen University.</h4>
### Introduction
This repository is for our TPAMI 2021 paper "[Adaptive Action Assessment](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9609694)"

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
