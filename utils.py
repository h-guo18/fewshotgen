import logging
import torch
import random
import numpy as np
import torch.nn.functional as F
import requests
RETRIEVE_API = 'http://192.168.1.99:8085/retrieve'

def set_logger(log_path):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


def set_random_seed(seed, cuda):
    """
    设置训练的随机种子
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < torch.topk(logits, top_k)[
            0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[...,
                                 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def retrieve_reference(context_list,k=3):
    results = []
    for context in context_list:
        r = requests.get(RETRIEVE_API,params = {'query':context,'k':k})
        neighbors = r.json()['neighbors']
        reference = '[SEP]'.join([ n['neighbor']+n['continuation']for n in neighbors])
        results.append(reference)
    return results

#UNIT TEST
if __name__ == '__main__':
    context_list = ['十八大以来，中共中央总书记习近平指出，中国共产党要紧紧依靠中国人民',
                    '人滑第三名日本选手村主章枝，冰上舞蹈世界排名第三的俄罗斯选手纳夫卡／科斯特马洛夫等一批世界级名将。\n俄罗斯的盐湖城冬奥会男子单人滑冠军亚古金因伤病原因退出了本次比赛，美国著名女子单人滑选手关颖珊因没有',
                    '清华大学建校100周年，校长邱勇与今天早上凌晨六点在主楼前面向全校师生发表重要讲话，鼓励大家自强不息厚德载物，一起打赢疫情防控清华保卫战'
                    ]
    print(retrieve_reference(context_list))
