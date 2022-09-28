from transformers import AutoModelForCausalLM,AutoTokenizer,TextGenerationPipeline
import json
import argparse
import os
import torch
from tqdm import tqdm
import rouge_zh



def gen(context):
    r= text_generator(context, max_length=400, do_sample=True, top_p=0.9, pad_token_id=50256,clean_up_tokenization_spaces =True)
    return(r[0]['generated_text'].replace(' ',''))
    
if __name__ == '__main__':
    device = 'cuda:0'
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='gongwen', help='few shot domain')
    parser.add_argument('--shotnum', type=int, default='128', help='number of samples during training')
    parser.add_argument('--model_dir', type=str, default='save/', help='model path')
    parser.add_argument('--adaption_type', type=str, default='finetune', help='finetune, adapter, or lora')
    parser.add_argument('--save_dir', type=str, default='./pred/', help='prediction file dump path')
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained('uer/gpt2-chinese-cluecorpussmall')
    if args.shotnum == 0:#zero-shot use original model
        model_path = 'uer/gpt2-chinese-cluecorpussmall'
    else:
        model_list = os.listdir(args.model_dir)
        model_paths = []
        for each in model_list:
            if f'{args.domain}_{args.adaption_type}_{args.shotnum}epoch' in each:
                model_paths.append(each)
        if len(model_paths) == 0:#no such model,raise error
            print(f'Error: no model find: {args.domain}_{args.adaption_type}_{args.shotnum}')
            model_path = None
        else:
            if len(model_paths) > 1 :#multiple model, use first one.
                print(f'Multiple model find: {model_list},will use first one.')
            model_path = os.path.join(args.model_dir,model_paths[0])
    if model_path:
        print(f'=====test model:{model_path}=====')
        if args.adaption_type != 'finetune':
            model = AutoModelForCausalLM.from_pretrained('uer/gpt2-chinese-cluecorpussmall').to(device)
            model.load_state_dict(torch.load(os.path.join(model_path,"delta.ckpt")), strict=False)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        text_generator = TextGenerationPipeline(model, tokenizer,device=0)


        with open(f'data/{args.domain}/{args.domain}_test.json')as f:
            datas = json.loads(f.read())
        samples = []
        for data in datas[:30]:
            all_context = data['content']
            start = 0
            while start + 200 < len(all_context):
                samples.append({'context':all_context[start:start+100],'ref':all_context[start+100:start+200]})
                start += 100
        print('=====test_samples[0]:=====')
        print(samples[0])

        preds = []
        for sample in tqdm(samples):
            preds.append({'context':sample['context'],
                        'ref':sample['ref'],
                        'pred':gen(sample['context'])
                            })
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        with open(os.path.join(args.save_dir, f'pred_{os.path.basename(model_path).replace(".json","")}.json' if args.shotnum != 0 else f'pred_{args.domain}_0shot.json'),'w')as f:
            f.write(json.dumps(preds,ensure_ascii=False))
        print('prediction file saved.')


