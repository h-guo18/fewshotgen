import argparse
import math
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import logging
from datetime import datetime
import os
from torch.utils.data import Dataset, DataLoader
from os.path import join, exists
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch.nn import DataParallel
import transformers
import pickle
import sys
from utils import set_logger, set_random_seed
from sklearn.model_selection import train_test_split
from data_parallel import BalancedDataParallel
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch.nn.utils.rnn as rnn_utils
import numpy as np
from dataset import CPMDataset
import json
from tensorboardX import SummaryWriter
from opendelta import Visualization, AdapterModel
import copy
TEST_DOMAIN = ['gongwen','international','poetry','sports','story']


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', default='', type=str, required=True, help='adaption domain')
    parser.add_argument('--shotnum', default=128, type=int, required=True, help='fewshot shotnum')
    parser.add_argument('--adaption_type', type=str, default='finetune', help='finetune,adapter,or lora')

    parser.add_argument('--device', default='0', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行训练')
    parser.add_argument('--vocab_path', default='vocab/chinese_vocab.model', type=str, required=False,
                        help='sp模型路径')
    parser.add_argument('--model_config', default='config/cpm-small.json', type=str, required=False,
                        help='需要从头训练一个模型时，模型参数的配置文件')
    parser.add_argument('--max_len', default=200, type=int, required=False, help='训练时，输入数据的最大长度')
    parser.add_argument('--log_path', default='log/train.log', type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--ignore_index', default=-100, type=int, required=False, help='对于ignore_index的label token不计算梯度')
    parser.add_argument('--batch_size', default=2, type=int, required=False, help='训练的batch size')
    parser.add_argument('--gpu0_bsz', default=6, type=int, required=False, help='0号卡的batch size')
    parser.add_argument('--lr', default=1e-5, type=float, required=False, help='学习率')
    parser.add_argument('--eps', default=1.0e-09, type=float, required=False, help='AdamW优化器的衰减率')
    parser.add_argument('--gradient_accumulation_steps', default=6, type=int, required=False, help='梯度积累的步数')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--save_model_path', default='save/', type=str, required=False,
                        help='模型输出路径')
    parser.add_argument('--pretrained_model', default='uer/gpt2-chinese-cluecorpussmall', type=str, required=False,
                        help='预训练的模型的路径')
    parser.add_argument('--seed', type=int, default=1234, help='设置随机种子')
    parser.add_argument('--num_workers', type=int, default=0, help="dataloader加载数据时使用的线程数量")
    parser.add_argument('--patience', type=int, default=10, help="用于early stopping,设为0时,不进行early stopping.early stop得到的模型的生成效果不一定会更好。")
    # parser.add_argument('--label_smoothing', default=True, action='store_true', help='是否进行标签平滑')
    parser.add_argument('--model_name', type=str, default='', help='model name')
    # parser.add_argument('--dev_size', type=int, default='200', help='number of samples in dev set')

    args = parser.parse_args()
    return args


def collate_fn(batch):
    input_ids = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=5)
    labels = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=-100)
    return input_ids, labels


def load_dataset(logger, args):
    """
    加载训练集
    """
    logger.info("loading training dataset")
    train_path = args.train_path
    dev_path = args.dev_path

    with open(train_path, "r") as f:
        train_list = json.loads(f.read())
    with open(dev_path, "r") as f:
        dev_list = json.loads(f.read())

    # test
    # train_list = train_list[:24]

    train_dataset = CPMDataset(train_list, args.max_len)
    dev_dataset = CPMDataset(dev_list[:len(train_list)], args.max_len) #dev set should be the same size as train set. note that it's not args.shotnum.

    return train_dataset, dev_dataset


def train_epoch(model, train_dataloader, optimizer, scheduler, logger,
                epoch, args,writer):
    model.train()
    device = args.device
    ignore_index = args.ignore_index
    epoch_start_time = datetime.now()

    total_loss = 0  # 记录下整个epoch的loss的总和
    epoch_correct_num = 0   # 每个epoch中,预测正确的word的数量
    epoch_total_num = 0  # 每个epoch中,预测的word的总数量
    batch_loss =0
    pbar = tqdm(enumerate(train_dataloader),total = len(train_dataloader),desc = f'epoch {epoch}/ max:{args.max_epochs}')


    for batch_idx, (input_ids, labels) in pbar:
        # 捕获cuda out of memory exception
        try:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model.forward(input_ids, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
            loss = loss.mean()

            # 统计该batch的预测token的正确数与总数
            batch_correct_num, batch_total_num = calculate_acc(logits, labels, ignore_index=ignore_index)
            # 统计该epoch的预测token的正确数与总数
            epoch_correct_num += batch_correct_num
            epoch_total_num += batch_total_num
            # 计算该batch的accuracy
            batch_acc = batch_correct_num / batch_total_num

            total_loss += loss.item()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # 进行一定step的梯度累计之后，更新参数
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                # 更新参数
                optimizer.step()
                # 更新学习率
                scheduler.step()
                # 清空梯度信息
                optimizer.zero_grad()
            pbar.set_postfix({'batch_loss':loss.item() * args.gradient_accumulation_steps, 'lr':scheduler.get_lr()})

            del input_ids, outputs

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                logger.info("WARNING: ran out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logger.info(str(exception))
                raise exception

    # 记录当前epoch的平均loss与accuracy
    epoch_mean_loss = total_loss / len(train_dataloader)
    epoch_mean_acc = epoch_correct_num / epoch_total_num
    writer.add_scalar('epoch_loss', epoch_mean_loss,global_step =epoch + 1)
    writer.add_scalar('epoch_acc', epoch_mean_acc,global_step =epoch + 1)
    logger.info(
        "epoch {}: loss {}, predict_acc {}".format(epoch + 1, epoch_mean_loss, epoch_mean_acc))

    logger.info('epoch {} finished'.format(epoch + 1))
    epoch_finish_time = datetime.now()
    logger.info('time for one epoch: {}'.format(epoch_finish_time - epoch_start_time))

    return epoch_mean_loss

def validation_epoch(model, dev_dataloader, logger,
                epoch, args,writer):
    model.eval()
    device = args.device
    ignore_index = args.ignore_index
    epoch_start_time = datetime.now()

    total_loss = 0  # 记录下整个epoch的loss的总和
    epoch_correct_num = 0   # 每个epoch中,预测正确的word的数量
    epoch_total_num = 0  # 每个epoch中,预测的word的总数量
    pbar = tqdm(enumerate(dev_dataloader),total = len(dev_dataloader),desc = f'validation for epoch {epoch}/ max:{args.max_epochs}')

    with torch.no_grad():
        for batch_idx, (input_ids, labels) in pbar:
            # 捕获cuda out of memory exception
            try:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                outputs = model.forward(input_ids, labels=labels)
                logits = outputs.logits
                loss = outputs.loss
                loss = loss.mean()

                # 统计该batch的预测token的正确数与总数
                batch_correct_num, batch_total_num = calculate_acc(logits, labels, ignore_index=ignore_index)
                # 统计该epoch的预测token的正确数与总数
                epoch_correct_num += batch_correct_num
                epoch_total_num += batch_total_num
                total_loss += loss.item()
                pbar.set_postfix({'loss':loss})
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                del input_ids, outputs

            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    logger.info("WARNING: ran out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    logger.info(str(exception))
                    raise exception

    # 记录当前epoch的平均loss与accuracy
    epoch_mean_loss = total_loss / len(dev_dataloader)
    epoch_mean_acc = epoch_correct_num / epoch_total_num
    writer.add_scalar('valid_loss', epoch_mean_loss,global_step =epoch + 1)
    writer.add_scalar('valid_acc', epoch_mean_acc,global_step =epoch + 1)
    logger.info(
        "validation for epoch {}: loss {}, predict_acc {}".format(epoch + 1, epoch_mean_loss, epoch_mean_acc))
    epoch_finish_time = datetime.now()

    return epoch_mean_loss

def train_stop(args, valid_losses):
    """return whether or not stop training."""
    if len(valid_losses) >= args.patience:
        if min(valid_losses) not in valid_losses[-args.patience:]: #no better epoch, stop training
            return True
        else:
            return False
    else:
        return False

def train(model, logger, train_dataset, dev_dataset, args,writer):
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn,
        drop_last=True
    )
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn,
        drop_last=True
    )
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    optimizer = transformers.AdamW(model.parameters(), lr=args.lr, eps=args.eps)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0.6*t_total, num_training_steps=t_total
    )

    logger.info('start training')

    train_losses = []   # 记录每个epoch的平均loss
    valid_losses = []
    # ========== start training ========== #
    for epoch in range(args.max_epochs):
        train_loss = train_epoch(
            model=model, train_dataloader=train_dataloader,
            optimizer=optimizer, scheduler=scheduler,
            logger=logger, epoch=epoch, args=args,writer=writer)
        train_losses.append(round(train_loss, 4))
        # logger.info("train loss list:{}".format(train_losses))
        #validation
        valid_loss = validation_epoch(
            model=model, dev_dataloader=dev_dataloader,
            logger=logger, epoch=epoch, args=args,writer=writer)
        if len(valid_losses) > 0 and valid_loss <= min(valid_losses):
            best_epoch = epoch
            best_model = copy.deepcopy(model)
        valid_losses.append(round(valid_loss, 4))
        if train_stop(args,valid_losses):
            break
    # save model
    logger.info('saving model for epoch {}'.format(best_epoch + 1))
    model_path = join(args.save_model_path,f'{args.model_name}epoch{best_epoch + 1}')
    if not os.path.exists(model_path):
        os.mkdir(model_path)
        
    if args.adaption_type != 'finetune':
        torch.save(best_model.state_dict(), join(model_path,"delta.ckpt"))
    else:
        best_model.save_pretrained(model_path)

    logger.info('training finished')
    logger.info("train_losses:{}".format(train_losses))
    logger.info("valid_losses:{}".format(valid_losses))


def caculate_loss(logit, target, pad_idx, smoothing=True):
    if smoothing:
        logit = logit[..., :-1, :].contiguous().view(-1, logit.size(2))
        target = target[..., 1:].contiguous().view(-1)

        eps = 0.1
        n_class = logit.size(-1)

        one_hot = torch.zeros_like(logit).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(logit, dim=1)

        non_pad_mask = target.ne(pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).mean()  # average later
    else:
        # loss = F.cross_entropy(predict_logit, target, ignore_index=pad_idx)
        logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
        labels = target[..., 1:].contiguous().view(-1)
        loss = F.cross_entropy(logit, labels, ignore_index=pad_idx)
    return loss


def calculate_acc(logit, labels, ignore_index=-100):
    logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
    labels = labels[..., 1:].contiguous().view(-1)

    _, logit = logit.max(dim=-1)  # 对于每条数据，返回最大的index
    # 进行非运算，返回一个tensor，若labels的第i个位置为pad_id，则置为0，否则为1
    non_pad_mask = labels.ne(ignore_index)
    n_correct = logit.eq(labels).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    return n_correct, n_word


def main():
    # 初始化参数
    args = set_args()
    args.max_epochs = int(12800 / args.shotnum) # this is sufficent. real epoch num is decided by tarin_stop() function.
    args.model_name = f'{args.domain}_{args.adaption_type}_{args.shotnum}'
    args.train_path = f'data/{args.domain}/preprocessed/{args.domain}_train_{args.shotnum}.json'
    args.dev_path = f'data/{args.domain}/preprocessed/{args.domain}_dev.json'
    print(f'=====MODEL NAME:{args.model_name}=====')
    writer = SummaryWriter(comment=args.model_name + args.train_path)


    args.cuda = not args.no_cuda

    # if args.batch_size < 2048 and args.warmup_steps <= 4000:
    #     print('[Warning] The warmup steps may be not enough.\n' \
    #           '(sz_b, warmup) = (2048, 4000) is the official setting.\n' \
    #           'Using smaller batch w/o longer warmup may cause ' \
    #           'the warmup stage ends with only little data trained.')

    # 创建日志对象
    logger = set_logger(args.log_path)
    # 当用户使用GPU,并且GPU可用时
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda:0' if args.cuda else 'cpu'
    args.device = device
    logger.info('using device:{}'.format(device))

    # 设置随机种子
    set_random_seed(args.seed, args.cuda)

    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    args.eod_id = tokenizer.convert_tokens_to_ids("<eod>")  # 文档结束符
    args.pad_id = tokenizer.pad_token_id

    # 创建模型的输出目录
    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)

    # 创建模型        
    model = AutoModelForCausalLM.from_pretrained(args.pretrained_model)
    model = model.to(device)
    logger.info('model config:\n{}'.format(model.config.to_json_string()))
    assert model.config.vocab_size == tokenizer.vocab_size
    if args.adaption_type:
        if args.adaption_type == 'adapter':
            delta_model = AdapterModel(model, modified_modules=['mlp','attn'],bottleneck_dim=12)
            delta_model.freeze_module(exclude=["deltas", "ln_f"], set_state_dict=True)
            delta_model.log()
        elif args.adaption_type == 'lora':
            delta_model = AdapterModel(model, modified_modules=['mlp','attn'],bottleneck_dim=12)
            delta_model.freeze_module(exclude=["deltas", "ln_f"], set_state_dict=True)
            delta_model.log()
            

    # 多卡并行训练模型
    if args.cuda and torch.cuda.device_count() > 1:
        # model = DataParallel(model).cuda()
        model = BalancedDataParallel(args.gpu0_bsz, model, dim=0).cuda()
        logger.info("use GPU {} to train".format(args.device))

    # 计算模型参数数量
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    logger.info('number of model parameters: {}'.format(num_parameters))

    # 记录参数设置
    logger.info("args:{}".format(args))

    # 加载训练集和验证集
    # ========= Loading Dataset ========= #
    train_dataset,dev_dataset = load_dataset(logger, args)

    train(model, logger, train_dataset, dev_dataset,args,writer)


if __name__ == '__main__':
    main()
