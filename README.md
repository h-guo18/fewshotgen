# fewshot generation  
This repository is for few-shot text generation research;  

## Task Description
**Given a handful of training samples, we explore to maximize the adaption performance for CausalLM task.**  
Specificlly, we use GPT-2 as backbone model and predict next 40 tokens given the previous 200 tokens. We also take the following practical requirements into consideration:  
1. **Parameter-efficiency**  
2. **Generalization ability to any new domain**  
3. **Stability in change of shotnums**

## Usage
### prepare data
Unzip `data.zip` to repository main directory.  
We provide corpus from 5 different domains: gongwen, international news, peotry, sports news, and short stories.

### enviroments  
`pip install -r requirements.txt`  

### train model  
`python train.py --shotnum $shotnum --domain $domain --adaption_type $adaption_type`  
And the model will be saved to `save/` directory by default.  
### test model  
`python test.py --shotnum $shotnum --domain $domain --adaption_type $adaption_type`  
And the prediction file will be saved to `pred/` directory by default.  
####  command arguments  
`$shotnum`: number of examples, possible values: {4,8,16,32,64,128};  
`$domain`: domain of adaption, {'gongwen', 'international', 'peotry', 'sports', 'story'};  
`$adaption_type`: 'finetune', 'adapter', or 'lora'; indicate type of adpation;  
