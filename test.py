from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline, BertModel
import json
import argparse
import os
import torch
from tqdm import tqdm
import rouge_zh
from utils import top_k_top_p_filtering, retrieve_reference
import torch.nn.functional as F


def generate_next_token(input_ids,ref_hidden_states):
    """
    对于给定的上文，生成下一个单词
    """
    # 只根据当前位置的前context_len个token进行生成
    input_ids = input_ids[:, -args.context_len:]

    outputs = model(input_ids=input_ids,
                    encoder_hidden_states=ref_hidden_states)
    logits = outputs.logits
    # next_token_logits表示最后一个token的hidden_state对应的prediction_scores,也就是模型要预测的下一个token的概率
    next_token_logits = logits[0, -1, :]
    next_token_logits = next_token_logits / args.temperature
    # 对于<unk>的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
    next_token_logits[unk_id] = -float('Inf')
    filtered_logits = top_k_top_p_filtering(
        next_token_logits, top_k=args.topk, top_p=args.topp)
    # torch.multinomial表示从候选集合中选出无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
    next_token_id = torch.multinomial(
        F.softmax(filtered_logits, dim=-1), num_samples=1)
    return next_token_id


def gen(context):

    input_ids = tokenizer.encode(context, add_special_tokens=False)

    cur_len = len(input_ids)
    last_token_id = input_ids[-1]  # 已生成的内容的最后一个token
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    # retrieved passage encodings if args.adaption_type == 'retrieval' else None
    if args.adaption_type == 'retrieval':
        references = retrieve_reference(tokenizer.batch_decode(input_ids))
        ref_ids = bert_tokenizer(
            references, padding=True, truncation=True, return_tensors='pt').to(device)
        ref_hidden_states = bert_model(**ref_ids).last_hidden_state
    else:
        references, ref_hidden_states = None,None

    while True:
        next_token_id = generate_next_token(input_ids,ref_hidden_states)
        input_ids = torch.cat((input_ids, next_token_id.unsqueeze(0)), dim=1)
        cur_len += 1
        word = tokenizer.convert_ids_to_tokens(next_token_id.item())
        # # 超过最大长度，并且换行
        # if cur_len >= max_len and last_token_id == 8 and next_token_id == 3:
        #     break
        # # 超过最大长度，并且生成标点符号
        # if cur_len >= max_len and word in [".", "。", "！", "!", "?", "？", ",", "，"]:
        #     break
        if cur_len >= args.max_len:
            break
        # 生成结束符
        if next_token_id == eod_id:
            break
    result = tokenizer.decode(input_ids.squeeze(
        0)).replace('[SEP]', '').replace(' ', '')
    return result,references


if __name__ == '__main__':
    device = 'cuda:0'
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str,
                        default='gongwen', help='few shot domain')
    parser.add_argument('--shotnum', type=int, default='128',
                        help='number of samples during training')
    parser.add_argument('--model_dir', type=str,
                        default='save/', help='model path')
    parser.add_argument('--adaption_type', type=str,
                        default='finetune', help='finetune, adapter, or lora')
    parser.add_argument('--save_dir', type=str,
                        default='./pred/', help='prediction file dump path')
    parser.add_argument('--context_len', default=200, type=int,
                        required=False, help='文本生成中，每一步生成时，参考的上文的长度')
    parser.add_argument('--max_len', default=400, type=int,
                        required=False, help='上文和生成文本加起来的最大长度')
    parser.add_argument('--temperature', default=1,
                        type=float, required=False, help='生成温度')
    parser.add_argument('--topk', default=0, type=int,
                        required=False, help='最高几选一')
    parser.add_argument('--topp', default=0.85, type=float,
                        required=False, help='最高积累概率')
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        'uer/gpt2-chinese-cluecorpussmall')
    eod_id = tokenizer.convert_tokens_to_ids("<eod>")  # 文档结束符
    sep_id = tokenizer.sep_token_id
    unk_id = tokenizer.unk_token_id
    if args.shotnum == 0:  # zero-shot use original model
        model_path = 'uer/gpt2-chinese-cluecorpussmall'
    else:
        model_list = os.listdir(args.model_dir)
        model_paths = []
        for each in model_list:
            if f'{args.domain}_{args.adaption_type}_{args.shotnum}epoch' in each:
                model_paths.append(each)
        if len(model_paths) == 0:  # no such model,raise error
            print(
                f'Error: no model find: {args.domain}_{args.adaption_type}_{args.shotnum}')
            model_path = None
        else:
            if len(model_paths) > 1:  # multiple model, use first one.
                print(f'Multiple model find: {model_list},will use first one.')
            model_path = os.path.join(args.model_dir, model_paths[0])
    if model_path:
        print(f'=====test model:{model_path}=====')
        if args.adaption_type not in ['finetune']:
            model = AutoModelForCausalLM.from_pretrained(
                'uer/gpt2-chinese-cluecorpussmall',
                add_cross_attention=(args.adaption_type == 'retrieval')).to(device)
            model.load_state_dict(torch.load(os.path.join(
                model_path, "delta.ckpt")), strict=False)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        print('MODEL lOADED')

        # 如果需要检索，加载额外的bert tokenizer
        if args.adaption_type == 'retrieval':
            bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
            bert_model = BertModel.from_pretrained(
                "bert-base-chinese").to(device)
            bert_model.eval()
        else:
            bert_tokenizer, bert_model = None, None
        print('BERT MODEL LOADED')

        with open(f'data/{args.domain}/{args.domain}_test.json')as f:
            datas = json.loads(f.read())
        samples = []
        # special treatment for gongwen domain
        testnum = 2 if args.domain == 'gongwen' else 100
        for data in datas[:testnum]:
            all_context = data['content']
            start = 0
            while start + 200 < len(all_context):
                samples.append(
                    {'context': all_context[start:start+100], 'ref': all_context[start+100:start+200]})
                start += 100

        results = []
        for sample in tqdm(samples):
            pred,retrieved_neighbours = gen(sample['context'])
            results.append({'context': sample['context'],
                            'ref': sample['ref'],
                            'pred': pred,
                            'retrieved_neighbours':retrieved_neighbours
                            })
            # print out first result
            if len(results) == 1:
                print('=====test_samples[0]:=====')
                print(results[0])
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        with open(os.path.join(args.save_dir, f'pred_{os.path.basename(model_path).replace(".json","")}.json' if args.shotnum != 0 else f'pred_{args.domain}_0shot.json'), 'w')as f:
            f.write(json.dumps(results, ensure_ascii=False))
        print('prediction file saved.')
