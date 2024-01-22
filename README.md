# DenoiseGNN
The code for IS2023 paper: A Graph Neural Network with Context Filtering and Feature Correction for Conversational emotion Recognition

# Requirements
* Python 3.7
* Pytorch 1.7+cu102
* Transformer(default version)

# Run steps
## Preprocess 
Method 1:
Please download datasets and .py file from [here](https://github.com/declare-lab/conv-emotion/tree/master/COSMIC/feature-extraction) and then preprocess the features on four dataset.

Method 2:
Please download datasets from official website. We did not make any additional deletions or modifications to the data. And then utilize the revised code we provided (from CoPMP) to fine-tine the RoBERTa-Large.

## Run
```bash
python run.py
```
To achieve the best performance, parameters \alpha and threshold need to be set to the values given in the paper.

# Citation
```
@article{gan2023graph,
  title={A Graph Neural Network with Context Filtering and Feature Correction for Conversational Emotion Recognition},
  author={Gan, Chenquan and Zheng, Jiahao and Zhu, Qingyi and Jain, Deepak Kumar and {\v{S}}truc, Vitomir},
  journal={Information Sciences},
  pages={120017},
  year={2023}
}
```
