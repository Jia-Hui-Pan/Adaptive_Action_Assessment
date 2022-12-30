# Adaptive_Action_Assessment
 Codes for the paper "Adaptive Action Assessment"

 Use Conda to install the environment. For example,
```
conda env create -f environment.yml
```

 And then activate the environment using
```
conda activate assessment
```


 Please follow the instructions for testing the model's performance in the AQA-7 dataset.
 1. Download the features to the "./feature" directory
 2. Download the checkpoints to the "./checkpoints"
 3. Run the evaluation code. 
```
python evaluation.py -gpu=0 -mode=structure -set-id=0 -loss-type=new_mse
#The '-set-id' can be set as 0, 1, 2, 3, 4 or 5, each representing one type of action in the AQA-7 dataset.
```
