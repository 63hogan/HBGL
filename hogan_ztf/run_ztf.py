from __future__ import absolute_import, division, print_function
from collections import defaultdict

import shutil
import argparse
import logging
import os
import json
import random
import re

import numpy as np

import torch
from torch.utils.data import (DataLoader, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from transformers import AdamW, get_linear_schedule_with_warmup
from torch import nn
import wandb
import tqdm

from ztf_s2s_ft.modeling import BertForSequenceToSequenceWithPseudoMask,BertForSequenceToSequence
from ztf_s2s_ft.modeling  import LabelSmoothingLoss

from transformers import AdamW, get_linear_schedule_with_warmup
# from transformers import BertConfig, BertTokenizer
from transformers import \
    RobertaConfig, BertConfig, \
    BertTokenizer, RobertaTokenizer
from transformers import BertForMaskedLM
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
    
from ztf_s2s_ft import utils
from ztf_s2s_ft.config import BertForSeq2SeqConfig
from collections import defaultdict

from data_tool import *

from torch.utils.tensorboard import SummaryWriter
import time

FORMAT = '%(asctime)s[%(name)s.%(funcName)s]%(levelname)s: %(message)s'


# 使用 basicConfig 应用格式
logging.basicConfig(level=logging.INFO, format=FORMAT)

MODEL_CLASSES = {
    'bert': (BertConfig, BertTokenizer),
    'roberta': (RobertaConfig, BertTokenizer),
}

def train_batch_labels(args, tokenizer, input_ids, attention_mask,  position_ids, _init_label_emb, label_child_set, label_parent_dic, hier_pos_id, hier_pos_emb_weight):

    
    config_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    model_config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None)
    config = BertForSeq2SeqConfig.from_exist_config(
        config=model_config, label_smoothing=args.label_smoothing,
        fix_word_embedding=args.fix_word_embedding,
        max_position_embeddings=args.max_source_seq_length + args.max_target_seq_length)

    logging.info("Model config for seq2seq: %s", str(config))

    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path)

    model_class = BertForSequenceToSequence 

    logging.info("Construct model %s" % model_class.MODEL_NAME)

    model = model_class.from_pretrained(
        args.model_name_or_path, config=config, model_type=args.model_type,
        # args.model_name_or_path, config=config, model_type='roberta',
        reuse_position_embedding=True,
        cache_dir=args.cache_dir if args.cache_dir else None)
    
    

    label_nums = input_ids.shape[0] - 2

    # model = BertForSequenceToSequence.from_pretrained(args.model_name_or_path)
    model = model.train()
    model.cuda()
    
    device = next(model.parameters()).device
    
    init_label_emb = _init_label_emb.float().cuda().requires_grad_()
    
    # hier_position_emb = hier_pos_emb.float().cuda().requires_grad_()

    hier_pos_emb_weight = hier_pos_emb_weight.to(device)
    # model.bert.embeddings.hier_position_embeddings = hier_position_emb
    model.bert.embeddings.hier_position_embeddings.weight.data = hier_pos_emb_weight
    model.bert.embeddings.hier_position_embeddings.weight.requires_grad_(True)

    optimizer_grouped_parameters = [
        {'params': [init_label_emb, ], 'weight_decay': 0.0},
        {"params": [model.bert.embeddings.hier_position_embeddings.weight], "weight_decay": 0.0}
    ]
    cpt_optimizer = AdamW(optimizer_grouped_parameters, lr=args.label_cpt_lr, eps=args.adam_epsilon)

    scaler = GradScaler(enabled=args.fp16)

    mask_ratio = 0.15
    bs = args.label_cpt_bsz
    b_input_ids = input_ids.unsqueeze(0).repeat(bs, 1).cuda().long()
    position_ids = position_ids.unsqueeze(0).repeat(bs, 1).cuda().long()
    hier_position_ids = hier_pos_id.unsqueeze(0).repeat(bs, 1).cuda().long()
    attention_mask = attention_mask.unsqueeze(0).repeat(bs, 1, 1).cuda().long()
    
    
    for step in range(args.label_cpt_steps):
        if args.label_cpt_not_incr_mask_ratio: #False
            c_mask_ratio = mask_ratio
        else:
            c_mask_ratio = mask_ratio + (step / args.label_cpt_steps) * 0.3
        inputs_embeds = torch.cat([model.bert.embeddings.word_embeddings.weight[tokenizer.cls_token_id].unsqueeze(0),
                                   init_label_emb,
                                   model.bert.embeddings.word_embeddings.weight[tokenizer.sep_token_id].unsqueeze(0),])
        
        inputs_embeds = inputs_embeds.unsqueeze(0).repeat(bs, 1, 1).cuda()
        
        mask_tokens = ~torch.bernoulli(torch.ones_like(b_input_ids) * (1 - c_mask_ratio)).bool()
        labels = torch.ones_like(b_input_ids).long() * -100
        # keep cls & sep unmask
        mask_tokens[:, 0] = 0
        mask_tokens[:, -1] = 0
        
        # labels[mask_tokens] = b_input_ids[mask_tokens] - model.bert.embeddings.word_embeddings.num_embeddings #从 0 开始
        
        masked_indices = torch.nonzero(mask_tokens, as_tuple=False)
        if masked_indices.numel() > 0:
            labels[masked_indices[:, 0], masked_indices[:, 1]] = masked_indices[:, 1] - 1
        
        inputs_embeds[mask_tokens] = model.bert.embeddings.word_embeddings.weight[tokenizer.mask_token_id]
        ##TODO set  target type id
        
        with autocast(enabled=args.fp16):
            outputs = model.bert(
                None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                token_type_ids=torch.ones_like(position_ids),
                hier_position_ids=hier_position_ids,
                
            )
            sequence_output = outputs[0]
            hidden_states = model.cls.predictions.transform(sequence_output)
            prediction_scores = hidden_states @ init_label_emb.T

            if args.label_cpt_use_bce: #True
                loss_fct = BCEWithLogitsLoss()  # -100 index = padding token
                with torch.no_grad():
                    bce_labels = torch.zeros_like(prediction_scores)
                    _bce_labels = []
                    for b in range(bs):
                        # l = labels[b][mask_tokens[b]].tolist()
                        # bce_l = bce_labels[b][mask_tokens[b]]
                        
                        b_mask_tokens_indices = torch.nonzero(mask_tokens[b], as_tuple=True)[0]
                        l = labels[b][b_mask_tokens_indices].tolist()
                        bce_l = bce_labels[b][b_mask_tokens_indices]
                        
                        
                        c = defaultdict(list)
                        lmap = {}
                        for il in l:
                            if il not in label_child_set: #叶子节点
                                # last labels
                                if il in label_parent_dic:
                                    p = label_parent_dic[il] # il 的父节点  
                                    c[p].append(il)
                                    lmap[il] = p
                        
                        for i, il in enumerate(l):
                            if il not in lmap:
                                bce_l[i][il] = 1
                            else:
                                for j in c[lmap[il]]:
                                    bce_l[i][j] = 1
                        _bce_labels.append(bce_l)
                    bce_labels = torch.cat(_bce_labels, dim=0)
                    # logging.info(bce_labels.sum())
                masked_lm_loss = loss_fct(prediction_scores[mask_tokens], bce_labels)
            else:
                loss_fct = CrossEntropyLoss()  # -100 index = padding token
                masked_lm_loss = loss_fct(prediction_scores.view(-1, label_nums), labels.view(-1))
        
        # #DEBUG
        # if step > 2:
        #     break

        scaler.scale(masked_lm_loss).backward()
        scaler.step(cpt_optimizer)
        scaler.update()
        

        init_label_emb.grad = None
        logging.info("step %d, masked_lm_loss: %f", step, masked_lm_loss.item())
    hier_pos_emb_weight = model.bert.embeddings.hier_position_embeddings.weight.data.clone().detach().cpu()
    del model
    del cpt_optimizer
    torch.cuda.empty_cache()
    
    return init_label_emb.detach().cpu(), hier_pos_emb_weight


def train_label_name_embedding(args):
    config_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    model_config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None)
    config = BertForSeq2SeqConfig.from_exist_config(
        config=model_config, label_smoothing=args.label_smoothing,
        fix_word_embedding=args.fix_word_embedding,
        max_position_embeddings=args.max_source_seq_length + args.max_target_seq_length)

    logging.info("Model config for seq2seq: %s", str(config))

    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path)

    model_class = BertForSequenceToSequenceWithPseudoMask 

    logging.info("Construct model %s" % model_class.MODEL_NAME)

    model = model_class.from_pretrained(
        args.model_name_or_path, config=config, model_type=args.model_type,
        # args.model_name_or_path, config=config, model_type='roberta',
        reuse_position_embedding=True,
        cache_dir=args.cache_dir if args.cache_dir else None)

    label_emb_cache_path = args.output_dir + '/label_name_emb_after_train.pt'
    hier_pos_emb_cache_path = args.output_dir + '/hier_pos_emb_after_train.pt'
    
    if args.add_vocab_file:
        init_label_emb = None
        ztf_map = read_key_value_file(args.ztf_path)
        label_name_set = get_uniq_train_cls_from_json(args.train_data_path)
        _, label_name_parent_dic = get_ztf_hierarchy_info(ztf_map)
        label_name_child_hiera_set, _ = get_train_labels_hiera_info(label_name_set,ztf_map)
        label_idx_dic = get_label_idx(label_name_child_hiera_set)
        
        label_tokens_start_index  = model.bert.embeddings.word_embeddings.num_embeddings
        
        label_tokens = [i for i in range(len(label_idx_dic))]
        for label in sorted(label_idx_dic.keys(),key=label_idx_dic.get):
            token = '__'+ label+ '__'
            tokenizer.add_tokens([token.lower()])
            
        hier_pos_emb = nn.Embedding(10, model.config.hidden_size)
        nn.init.xavier_uniform_(hier_pos_emb.weight)
        

        if args.load_label_embedding_cache and os.path.exists(label_emb_cache_path):
            logging.info("直接从本地加载 label embedding: %s", label_emb_cache_path)
            init_label_emb = torch.load(label_emb_cache_path)
            hier_pos_emb = torch.load(hier_pos_emb_cache_path)
        else:
                            
            if args.label_cpt:
                
                label_name_tensors = []
                max_l = -1

                label_name_cnt = check_cnt(label_name_child_hiera_set)
                assert(label_name_cnt == len(label_idx_dic))


                for label_name in sorted(label_idx_dic.keys(), key=label_idx_dic.get):
                    parents = []
                    l = label_name
                    while label_name_parent_dic[l] != 'root':
                        parents.append(label_name_parent_dic[l])
                        l = label_name_parent_dic[l]
                    parents.reverse()
                    parents.append(label_name)
                    label_str = ''
                    for l in parents:
                        label_str += ztf_map[l] + ' '
                    label_str = label_str.strip()
                    label_name_tensors.append(tokenizer.encode(label_str, add_special_tokens=False))
                    max_l = max(len(label_name_tensors[-1]), max_l)
                label_name_tensors = torch.LongTensor([i + [tokenizer.pad_token_id] * (max_l - len(i)) for i in label_name_tensors])
                
                with torch.no_grad():
                    init_label_emb = model.bert.embeddings.word_embeddings(label_name_tensors)
                    label_mask = label_name_tensors != tokenizer.pad_token_id
                    init_label_emb = (label_mask.unsqueeze(-1) * init_label_emb).sum(1) / label_mask.sum(1, keepdim=True).clamp(min=1)
                def _loop(a):
                    if label_name_parent_dic[a] != 'root':
                        return [a,] + _loop(label_name_parent_dic[a])
                    else:
                        return [a]
                label_level_dic = {} #每个 labal 的level, root 0
                for i in label_idx_dic.keys():
                    label_level_dic[i] = len(_loop(i))
        
                max_labels_in_batch = 500 
                trained_label_emb = torch.zeros_like(init_label_emb)
                
                label_name_batchs = get_labels_batch(label_name_child_hiera_set)
                
                trained_label_emb = torch.zeros_like(init_label_emb)
            
                batch_idx = 1
                for batch_labels_keys in label_name_batchs:
                    batch_original_indices = [label_idx_dic[name] for name in batch_labels_keys]
                    logging.info(f"start training label name batch idx:{batch_idx}")
                    if not batch_labels_keys:
                        continue
                    label_name_to_batch_idx = {name: j for j, name in enumerate(batch_labels_keys)}
                    
                    batch_init_label_emb = init_label_emb[batch_original_indices]
                    batch_attention_mask = torch.zeros((len(batch_labels_keys) + 2, len(batch_labels_keys) + 2))
                    batch_label_child_set = defaultdict(set)
                    batch_label_parent_dic = {}
                    
                    def _label_map_f_batch(label_name):
                        if label_name == 'root': return -1
                        return label_name_to_batch_idx.get(label_name, -999)
                    
                    for parent_label, child_labels in label_name_child_hiera_set.items():
                        for child_label in list(child_labels):
                            if parent_label in label_name_to_batch_idx and child_label in label_name_to_batch_idx:
                                parent_batch_idx = _label_map_f_batch(parent_label)
                                child_batch_idx = _label_map_f_batch(child_label)
                                batch_attention_mask[parent_batch_idx + 1][child_batch_idx + 1] = 1
                                batch_label_child_set[parent_batch_idx].add(child_batch_idx)
                                batch_label_parent_dic[child_batch_idx] = parent_batch_idx
                                batch_attention_mask[child_batch_idx + 1][parent_batch_idx + 1] = 1
                    # 自注意力
                    for i in range(len(batch_labels_keys)):
                        batch_attention_mask[i + 1, i + 1] = 1 
                    # [CLS] 和 [SEP] 也应能自注意
                    batch_attention_mask[0, 0] = 1
                    batch_attention_mask[-1, -1] = 1
                    
                    batch_label_tokens = ['__'+name+'__' for name in batch_labels_keys]
                    batch_input_ids_str = ' '.join(batch_label_tokens)
                    encoded_tokens = tokenizer.encode(batch_input_ids_str, add_special_tokens=False)
                    batch_input_ids = torch.LongTensor([tokenizer.cls_token_id] + encoded_tokens + [tokenizer.sep_token_id])
                    
                    assert len(batch_input_ids) == len(batch_labels_keys) + 2
                    # batch_label_classes = [label_level_dic[name] for name in batch_labels_keys]
                    # max_level_in_batch = max(batch_label_classes) if batch_label_classes else 0
                    # batch_position_ids = torch.LongTensor([0] + batch_label_classes + [0])
                    batch_label_classes = [label_level_dic[name]-1 for name in batch_labels_keys]
                    max_level_in_batch = max(batch_label_classes) if batch_label_classes else 0
                    batch_hier_pos_ids = torch.LongTensor([-1] + batch_label_classes + [-1])
                    batch_position_ids = torch.zeros_like(batch_hier_pos_ids)

                    hier_pos_emb_weight = hier_pos_emb.weight.data
                    
                    updated_batch_emb, hier_pos_emb_weight = train_batch_labels(
                        args, tokenizer, batch_input_ids, batch_attention_mask,
                        batch_position_ids, batch_init_label_emb, 
                        batch_label_child_set, batch_label_parent_dic,batch_hier_pos_ids,hier_pos_emb_weight
                    )
                    trained_label_emb[batch_original_indices] = updated_batch_emb
                    batch_idx += 1
                    # # TODO debug
                    # if batch_idx > 2:
                    #     break
            
                init_label_emb = trained_label_emb.detach().cpu()
                # hier_pos_emb = hier_pos_emb.detach().cpu()
                torch.save(init_label_emb, label_emb_cache_path)
                torch.save(hier_pos_emb, hier_pos_emb_cache_path)
        
        model.bert.embeddings.word_embeddings.weight.data = torch.cat([model.bert.embeddings.word_embeddings.weight.data, init_label_emb], dim=0)
        model.bert.embeddings.word_embeddings.num_embeddings += len(label_tokens)
        model.bert.embeddings.hier_position_embeddings = hier_pos_emb
        model.cls.predictions.bias.data =  torch.cat([model.cls.predictions.bias.data, torch.zeros(len(label_tokens))],
                                                        dim=0)
        original_vs = config.vocab_size
        config.vocab_size = config.vocab_size + len(label_tokens)
        logging.info(f"Re-initializing loss function with new vocab size: {config.vocab_size}")
        if config.label_smoothing > 0:
            model.crit_mask_lm_smoothed = LabelSmoothingLoss(
                config.label_smoothing, config.vocab_size, ignore_index=0, reduction='none')
            model.crit_mask_lm = None
        else:
            model.crit_mask_lm_smoothed = None
            model.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
        if args.softmax_label_only:
            model.label_start_index = label_tokens_start_index
    else:
        original_vs = config.vocab_size

    # if args.soft_label:
    #     model.soft_label = True
    #     model.mask_token_id = tokenizer.mask_token_id
    #     model.sep_token_id = tokenizer.sep_token_id
    #     model.vs = vs

    return model, tokenizer, original_vs

def prepare_for_training(args, model, checkpoint_state_dict ):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if checkpoint_state_dict:
        optimizer.load_state_dict(checkpoint_state_dict['optimizer'])
        model.load_state_dict(checkpoint_state_dict['model'])

    return model, optimizer


def train(args, training_features, model, tokenizer):
    """ Train the model """
    
    
    tensorboard_logdir = f"/tensor_log/my_experiment_{int(time.time())}"
    tensorboard_logdir = args.output_dir + tensorboard_logdir
    
    writer = SummaryWriter(tensorboard_logdir)

    logging.info(f"TensorBoard 日志将保存在: {tensorboard_logdir}")
    save_path = None

    # model recover
    recover_step = utils.get_max_epoch_model(args.output_dir)
    if recover_step :
        save_path = os.path.join(args.output_dir, "ckpt-%d" % recover_step) 

    if recover_step:
        checkpoint_state_dict = utils.get_checkpoint_state_dict(args.output_dir, recover_step)
    else:
        checkpoint_state_dict = None

    model.to(args.device)
    model, optimizer = prepare_for_training(args, model, checkpoint_state_dict)

    scaler = GradScaler(enabled=args.fp16)
    if checkpoint_state_dict and 'scaler' in checkpoint_state_dict:
        scaler.load_state_dict(checkpoint_state_dict['scaler'])
        
    per_node_train_batch_size = args.per_gpu_train_batch_size * args.n_gpu * args.gradient_accumulation_steps
    train_batch_size = per_node_train_batch_size * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    global_step = recover_step if recover_step else 0

    if args.num_training_steps == -1:
        args.num_training_steps = args.num_training_epochs * len(training_features) // train_batch_size

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_training_steps, last_epoch=-1)

    if checkpoint_state_dict:
        scheduler.load_state_dict(checkpoint_state_dict["lr_scheduler"])

    train_dataset = utils.Seq2seqDatasetForBert(
        features=training_features, max_source_len=args.max_source_seq_length,
        max_target_len=args.max_target_seq_length, vocab_size=model.bert.embeddings.word_embeddings.num_embeddings,
        cls_id=tokenizer.cls_token_id, sep_id=tokenizer.sep_token_id, pad_id=tokenizer.pad_token_id,
        mask_id=tokenizer.mask_token_id, random_prob=args.random_prob, keep_prob=args.keep_prob,
        offset=train_batch_size * global_step, num_training_instances=train_batch_size * args.num_training_steps,
        source_mask_prob=args.source_mask_prob, target_mask_prob=args.target_mask_prob,
        mask_way=args.mask_way, num_max_mask_token=args.num_max_mask_token,
    )


    logging.info("Check dataset:")
    for i in range(5):
        source_ids, target_ids = train_dataset.__getitem__(i)[:2]
        logging.info("Instance-%d" % i)
        logging.info("Source tokens = %s" % " ".join(tokenizer.convert_ids_to_tokens(source_ids)))
        logging.info("Target tokens = %s" % " ".join(tokenizer.convert_ids_to_tokens(target_ids)))
    
    logging.info("Mode = %s" % str(model))

    # Train!
    logging.info("  ***** Running training *****  *")
    logging.info("  Num examples = %d", len(training_features))
    logging.info(f" Num train_dataset  = {len(train_dataset)}")
    logging.info("  Num num_training_steps  = %d", args.num_training_steps)
    logging.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logging.info("  Batch size per node = %d", per_node_train_batch_size)
    logging.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", train_batch_size)
    logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logging.info("  Total optimization steps = %d", args.num_training_steps)

    if args.num_training_steps <= global_step:
        logging.info("Training is done. Please use a new dir or clean this dir!")
    else:
        # The training features are shuffled
        train_sampler = SequentialSampler(train_dataset) \
            if args.local_rank == -1 else DistributedSampler(train_dataset, shuffle=False)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler,
            batch_size=per_node_train_batch_size // args.gradient_accumulation_steps,
            collate_fn=utils.batch_list_to_batch_tensors)

        train_iterator = tqdm.tqdm(
            train_dataloader, initial=global_step * args.gradient_accumulation_steps,
            desc="Iter (loss=X.XXX, lr=X.XXXXXXX)", disable=args.local_rank not in [-1, 0])

        model.train()
        model.zero_grad()

        tr_loss, logging_loss = 0.0, 0.0
        for step, batch in enumerate(train_iterator):
            if global_step > args.num_training_steps:
                break
            batch = tuple(t.to(args.device) for t in batch)
            if args.mask_way == 'v2':
                inputs = {'source_ids': batch[0],
                        'target_ids': batch[1],
                        'label_ids': batch[2],
                        'pseudo_ids': batch[3],
                        'num_source_tokens': batch[4],
                        'num_target_tokens': batch[5]}
            elif args.mask_way == 'v1' or args.mask_way == 'v0':
                inputs = {'source_ids': batch[0],
                        'target_ids': batch[1],
                        'masked_ids': batch[2],
                        'masked_pos': batch[3],
                        'masked_weight': batch[4],
                        'num_source_tokens': batch[5],
                        'num_target_tokens': batch[6]}

            with autocast(enabled=args.fp16):
                loss = model(**inputs)
            # loss = model(**inputs)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training

            train_iterator.set_description('Iter (loss=%5.3f) lr=%9.7f' % (loss.item(), scheduler.get_lr()[0]))
            logging.info('Iter (loss=%5.3f) lr=%9.7f' % (loss.item(), scheduler.get_lr()[0]))
            if True:
                writer.add_scalar('train/loss', loss.item(), step)
                writer.add_scalar('train/learning_rate', scheduler.get_lr()[0],step)
                
                if (global_step + 1) % 50 == 0:
                    logging.info('global_step:%d (loss=%5.3f) lr=%9.7f' % (global_step, loss.item(), scheduler.get_lr()[0]))

            else:
                if (step + 1) % 50 == 0:
                    logging.info('train/loss', loss.item())

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()



            logging_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                model.zero_grad()
                
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging.info("")
                    logging.info(" Step [%d ~ %d]: %.2f", global_step - args.logging_steps, global_step, logging_loss)
                    logging_loss = 0.0

                if args.local_rank in [-1, 0] and args.save_steps > 0 and \
                        (global_step % args.save_steps == 0 or global_step == args.num_training_steps):

                    save_path = os.path.join(args.output_dir, "ckpt-%d" % global_step)
                    os.makedirs(save_path, exist_ok=True)
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(save_path)

                    optim_to_save = {
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": scheduler.state_dict(),
                    }
                    if args.fp16:
                        # optim_to_save["amp"] = amp.state_dict()
                        optim_to_save["scaler"] = scaler.state_dict()
                    torch.save(optim_to_save, os.path.join(save_path, utils.OPTIM_NAME))
                    logging.info("Saving model checkpoint %d into %s", global_step, save_path)

                    
    writer.close()
    return save_path


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="Training data (json format) for training. Keys: source and target")
    parser.add_argument("--valid_file", default=None, type=str, required=True,
                        help="Training data (json format) for training. Keys: source and target")
    parser.add_argument("--test_file", default=None, type=str,
                        help="Training data (json format) for training. Keys: source and target")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list:")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--log_dir", default=None, type=str,
                        help="The output directory where the log will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default=None, type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default=None, type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--max_source_seq_length", default=464, type=int,
                        help="The maximum total source sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_target_seq_length", default=48, type=int,
                        help="The maximum total target sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")

    parser.add_argument("--cached_train_features_file", default=None, type=str,
                        help="Cached training features file")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--label_smoothing", default=0.1, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_training_steps", default=-1, type=int,
                        help="set total number of training steps to perform")
    parser.add_argument("--num_training_epochs", default=20, type=int,
                        help="set total number of training epochs to perform (--num_training_steps has higher priority)")
    parser.add_argument("--num_warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--random_prob", default=0.1, type=float,
                        help="prob to random replace a masked token")
    parser.add_argument("--keep_prob", default=0.1, type=float,
                        help="prob to keep no change for a masked token")
    parser.add_argument("--fix_word_embedding", action='store_true',
                        help="Set word embedding no grad when finetuning.")

    parser.add_argument('--logging_steps', type=int, default=500,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

    parser.add_argument('--source_mask_prob', type=float, default=-1.0,
                        help="Probability to mask source sequence in fine-tuning")
    parser.add_argument('--target_mask_prob', type=float, default=0.5,
                        help="Probability to mask target sequence in fine-tuning")
    parser.add_argument('--num_max_mask_token', type=int, default=0,
                        help="The number of the max masked tokens in target sequence")
    parser.add_argument('--mask_way', type=str, default='v2',
                        help="Fine-tuning method (v0: position shift, v1: masked LM, v2: pseudo-masking)")
    parser.add_argument("--lmdb_cache", action='store_true',
                        help="Use LMDB to cache training features")
    parser.add_argument("--lmdb_dtype", type=str, default='h',
                        help="Data type for cached data type for LMDB")

    parser.add_argument("--add_vocab_file", type=str, default=None)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--softmax_label_only', action='store_true')

    parser.add_argument('--soft_label', action='store_true')
    parser.add_argument('--soft_label_hier_real', action='store_true')

    parser.add_argument('--one_by_one_label_init_map', type=str, default=None)
    parser.add_argument('--label_cpt', type=str, default=None)
    parser.add_argument('--label_cpt_lr', type=float, default=1e-3)
    parser.add_argument('--label_cpt_steps', type=int, default=500)
    parser.add_argument('--label_cpt_bsz', type=int, default=16)
    parser.add_argument('--label_cpt_not_incr_mask_ratio', action='store_true')
    parser.add_argument('--label_cpt_use_bce', action='store_true')

    parser.add_argument('--label_cpt_decodewithpos', action='store_true')

    parser.add_argument('--random_label_init', action='store_true')

    parser.add_argument('--nyt_only_last_label_init', action='store_true')

    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--only_test_path', type=str, default=None)

    parser.add_argument('--rcv1_expand', type=str, default=None)
    
    parser.add_argument('--ztf_path', type=str, default='/root/autodl-tmp/HBGL/data/ztfData/ztf/ztf_handle.txt')
    parser.add_argument('--train_data_path', type=str, default='/root/autodl-tmp/HBGL/data/ztfData/train/train_data_clear.jsonl')
    parser.add_argument('--load_label_embedding_cache', action='store_true')
    
    
    parser.add_argument
    args = parser.parse_args()
    return args


def prepare(args):
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        logging.info("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(
        args.output_dir, 'train_opt.json'), 'w'), sort_keys=True, indent=2)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logging.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    logging.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    # if args.fp16:
    #     try:
    #         import apex
    #         apex.amp.register_half_function(torch, 'einsum')
    #     except ImportError:
    #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")


def test(args, model_path):
    from test_ztf import main
    bout = None
    for i, save_path in enumerate(model_path):
        logging.info("start evaluating path:%s", save_path)
        if save_path is None: continue
        flags = ['--model_type'     , args.model_type                          ,
            '--tokenizer_name'         , args.model_name_or_path             ,
            '--input_file'             , args.test_file                  ,
            '--split'                  , 'test'                         ,
            '--do_lower_case'          ,
            '--model_path'             , str(save_path)              ,
            '--max_seq_length'         , str(args.max_source_seq_length + args.max_target_seq_length) if args.label_cpt_decodewithpos else str(args.max_source_seq_length)             ,
            '--max_tgt_length'         , str(args.max_target_seq_length)             ,
            '--batch_size'             , '128'                            ,
            '--beam_size'              , '1'                             ,
            '--length_penalty'         , '0'                             ,
            '--forbid_duplicate_ngrams',
            '--mode'                   , 's2s'                           ,
            '--forbid_ignore_word'     , '"."'                           ,
            '--cached_features_file'   , str(os.path.join(args.output_dir, "cached_features_for_test.pt")),
            '--add_vocab_file'         , args.add_vocab_file,
            '--ztf_path'               , args.ztf_path,
            '--train_data_path'        , args.train_data_path]

        if args.softmax_label_only:
            flags.append('--softmax_label_only')
        if args.soft_label:
            flags.append('--soft_label')
        if args.soft_label_hier_real:
            flags.append('--soft_label_hier_real_with_train_file')
            flags.append(args.train_file)
        if args.model_type == 'roberta':
            del flags[flags.index('--do_lower_case')]
        if args.label_cpt_decodewithpos:
            flags.append('--target_no_offset')

        out = main(flags)
        prefix = 'test' + 'micro' if i == 0 else 'macro'
        if args.wandb:
            wandb.log({f'{prefix}/macro_f1': out['macro_f1'], f'{prefix}/micro_f1': out['micro_f1']})
            if bout is None or bout['macro_f1'] < out['macro_f1']:
                bout = out

    if args.wandb and bout:
        prefix = 'test'
        wandb.log({f'{prefix}/macro_f1': bout['macro_f1'], f'{prefix}/micro_f1': bout['micro_f1']})

def get_chkpt_directories(root_dir="/root/autodl-tmp/HBGL/hogan_ztf/roberta_model/"):
    chkpt_dirs = []
    
    # 检查目录是否存在
    if not os.path.exists(root_dir):
        print(f"警告: 目录 {root_dir} 不存在")
        return chkpt_dirs
    
    if not os.path.isdir(root_dir):
        print(f"警告: {root_dir} 不是一个目录")
        return chkpt_dirs
    
    try:
        # 遍历目录中的所有项
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            
            # 检查是否是目录且名称匹配 chkpt-*
            if os.path.isdir(item_path) and re.match(r'ckpt-.*', item):
                chkpt_dirs.append(item_path)
    except PermissionError:
        print(f"错误: 没有权限访问目录 {root_dir}")
    except Exception as e:
        print(f"遍历目录时发生错误: {e}")
    return chkpt_dirs


def start_train():
    logging.info("start trainging.....")    
    args = get_args()
    prepare(args)
    logging.info(args)
    if args.only_test:
        args.wandb = False
        test(args, args.only_test_path, None)
        exit(0)

    if args.wandb:
        wandb.init(
            project="HBGL",
            name=args.output_dir.split('/')[-1],
        )
        wandb.define_metric("train/global_step")
        wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training will download model & vocab
    # Load pretrained model and tokenizer
    # train_label_name_embedding(args)
    # return
    model, tokenizer, vs = train_label_name_embedding(args)

    if args.local_rank == 0:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training will download model & vocab

    if args.cached_train_features_file is None:
        if not args.lmdb_cache:
            args.cached_train_features_file = os.path.join(args.output_dir, "cached_features_for_training.pt")
        else:
            args.cached_train_features_file = os.path.join(args.output_dir, "cached_features_for_training_lmdb")


    num_lines = sum(1 for line in open(args.train_file))
    training_features = utils.load_and_cache_examples(
        example_file=args.train_file, tokenizer=tokenizer, local_rank=args.local_rank,
        cached_features_file=args.cached_train_features_file, shuffle=True,
        lmdb_cache=args.lmdb_cache, lmdb_dtype=args.lmdb_dtype,
    )

    if args.add_vocab_file:
        for i in training_features:
            for j in i.target_ids:
                assert j >= vs

    save_path = train(args, training_features, model, tokenizer)
    # if args.test_file:
    #     test(args, save_path)
    return args

def test_main(save_path):
    logging.info("start test.....")
    args = get_args()
    logging.info(args)
    test(args, save_path)
    


def main():
    # args = start_train()
    save_path = get_chkpt_directories("/root/autodl-tmp/HBGL/hogan_ztf/roberta_model/hier_pos_lr_7e-5")
    logging.info(f"找到的检查点目录: {save_path}")
    test_main(save_path)

if __name__ == "__main__":
    main()
