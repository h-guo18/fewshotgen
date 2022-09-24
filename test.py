from transformers import AutoModelForCausalLM,AutoTokenizer,TextGenerationPipeline
import json
import argparse
import os
import torch



def gen(context):
    r= text_generator(context, max_length=400, do_sample=True, top_p=0.9, pad_token_id=50256,clean_up_tokenization_spaces =True)
    return(r[0]['generated_text'].replace(' ',''))
    
if __name__ == '__main__':
    device = 'cuda:0'
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='gongwen', help='few shot domain')
    parser.add_argument('--shotnum', type=int, default='128', help='number of samples during training')
    parser.add_argument('--epochs', type=str, default='-1', help='number of epoch during training')
    parser.add_argument('--model_dir', type=str, default='save/', help='model path')
    parser.add_argument('--adaption_type', type=str, default='finetune', help='finetune, adapter, or lora')
    parser.add_argument('--save_dir', type=str, default='./pred/', help='prediction file dump path')
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained('uer/gpt2-chinese-cluecorpussmall')
    args.epochs = int(1280 / args.shotnum) if args.shotnum != 0 else 0
    model_path = os.path.join(args.model_dir,f'{args.domain}_{args.adaption_type}_{args.shotnum}epoch{args.epochs}')
    print(f'=====test model:{model_path}=====')
    if args.shotnum == 0:
        model = AutoModelForCausalLM.from_pretrained('uer/gpt2-chinese-cluecorpussmall').to(device)
    else:
        if args.adaption_type != 'finetune':
            model = AutoModelForCausalLM.from_pretrained('uer/gpt2-chinese-cluecorpussmall').to(device)
            model.load_state_dict(torch.load(os.path.join(model_path,"delta.ckpt")), strict=False)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    text_generator = TextGenerationPipeline(model, tokenizer,device=0)


    with open(f'data/{args.domain}/{args.domain}_test.json')as f:
        datas = json.loads(f.read())
    samples = []
    for data in datas[:1]:
        all_context = data['content']
        start = 0
        while start + 200 < len(all_context):
            samples.append({'context':all_context[start:start+100],'ref':all_context[start+100:start+200]})
            start += 100
    print('=====test_samples[0]:=====')
    print(samples[0])

    from tqdm import tqdm
    preds = []
    for sample in tqdm(samples):
        preds.append({'context':sample['context'],
                      'ref':sample['ref'],
                      'pred':gen(sample['context'])
                        })
    with open(f'pred_{os.path.basename(model_path).replace(".json","")}.json','w')as f:
        f.write(json.dumps(preds,ensure_ascii=False))
    print('prediction file saved.')


