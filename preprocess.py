import argparse
from utils import set_logger
from transformers import AutoTokenizer
import os
import pickle
from tqdm import tqdm
import json


def preprocess():
    """
    对故事数据集进行预处理
    """
    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', default='log/preprocess.log', type=str, required=False, help='日志存放位置')
    parser.add_argument('--data_path', default='data/gongwen', type=str, required=False, help='数据集存放位置')
    parser.add_argument('--save_path', default='data/', type=str, required=False, help='对训练数据集进行tokenize之后的数据存放位置')
    parser.add_argument('--win_size', default=200, type=int, required=False, help='滑动窗口的大小，相当于每条数据的最大长度')
    parser.add_argument('--step', default=200, type=int, required=False, help='滑动窗口的滑动步幅')
    args = parser.parse_args()

    # 初始化日志对象
    logger = set_logger(args.log_path)

    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
    eod_id = tokenizer.convert_tokens_to_ids("<eod>")   # 文档结束符
    sep_id = tokenizer.sep_token_id

    # 读取作文数据集目录下的所有文件
    
    logger.info("start tokenizing data")
    for file in tqdm(os.listdir(args.data_path)):
        train_list = []
        try:
            file = os.path.join(args.data_path, file)
            with open(file, "r", encoding="utf8")as reader:
                passages = json.loads(reader.read())
                for passage in passages:

                    title = passage['title'].strip()    # 取出标题
                    article = passage['content']  # 取出正文内容
                    title_ids = tokenizer.encode(title, add_special_tokens=False)
                    article_ids = tokenizer.encode(article, add_special_tokens=False)
                    token_ids = title_ids + [sep_id] + article_ids + [eod_id]
                    # train_list.append(token_ids)

                    # 对于每条数据，使用滑动窗口对其进行截断
                    win_size = args.win_size
                    step = args.step
                    start_index = 0
                    end_index = win_size
                    data = token_ids[start_index:end_index]
                    train_list.append(data)
                    start_index += step
                    end_index += step
                    while end_index+50 < len(token_ids):  # 剩下的数据长度，大于或等于50，才加入训练数据集
                        data = token_ids[start_index:end_index]
                        train_list.append(data)
                        start_index += step
                        end_index += step

            # 序列化训练数据
            with open(os.path.join(args.data_path,'preprocessed/',file.split('/')[-1]), "w") as f:
                json.dump(train_list, f,ensure_ascii=False)
        except:
            pass


if __name__ == '__main__':
    preprocess()