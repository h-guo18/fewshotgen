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
`$shotnum`: number of examples, possible values: {0,4,8,16,32,64,128};  
`$domain`: domain of adaption, {'gongwen', 'international', 'peotry', 'sports', 'story'};  
`$adaption_type`: 'finetune', 'adapter', 'lora', or 'retrieval'; indicate methods of adpation to target domain;  
>* __'finetune'__: Traditional full-parameter adaption;  
>* __'adapter'__: Parameter-efficient tuning by adding parameter blocks, paper: <https://arxiv.org/pdf/1902.00751.pdf>;  
>* __'lora'__: Parameter-efficient tuning by adding low-rank matrics, paper: <https://arxiv.org/pdf/2106.09685.pdf>;  
>* __'retrieval'__: Input encodings of retrieved passages as reference. Training with this settings will add cross-attention blocks and freeze other parameters. The result should be a domain-agnostic LM with ability to consult given passages.  

## Results
#### BLEU  
![bleu](https://user-images.githubusercontent.com/67671475/194069362-6bb9872b-7ff1-4de4-83d3-80020de9358f.png)  
#### BERTScore  
![bertscore](https://user-images.githubusercontent.com/67671475/194069500-8cb65ae2-81c9-48d0-bd36-b8697448b1ef.png)  
#### Rouge-2  
![rouge-2](https://user-images.githubusercontent.com/67671475/194069598-e2921309-7060-40af-9dd9-d1cf3d8d46af.png)  


