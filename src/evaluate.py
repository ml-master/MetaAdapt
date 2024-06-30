import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from datetime import datetime

from roberta_utils import *
from dataloader import *
from dataset import get_dataset
from config import *
from utils import *
from tqdm import tqdm
import json
import os


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluation(args, model, eval_dataloader):
    eval_preds, eval_labels, eval_losses = [], [], []
    tqdm_dataloader = tqdm(eval_dataloader)
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, attention_mask, labels = batch     
            outputs = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                    )

            loss = outputs[0]
            logits = outputs[1]
            eval_preds += torch.argmax(logits, dim=1).cpu().numpy().tolist()
            eval_labels += labels.cpu().numpy().tolist()
            eval_losses.append(loss.item())

            tqdm_dataloader.set_description('Eval bacc: {:.4f}, acc: {:.4f}, f1: {:.4f}, loss: {:.4f}'.format(
                balanced_accuracy_score(eval_labels, eval_preds),
                np.mean(np.array(eval_labels)==np.array(eval_preds)), 
                f1_score(eval_labels, eval_preds),
                np.mean(eval_losses)
            ))

    final_bacc = balanced_accuracy_score(eval_labels, eval_preds)
    final_acc = np.mean(np.array(eval_preds)==np.array(eval_labels))
    final_f1 = f1_score(eval_labels, eval_preds)
    final_precision = precision_score(eval_labels, eval_preds)
    final_recall = recall_score(eval_labels, eval_preds)
    precisions = precision_score(eval_labels, eval_preds,average=None)
   
    return final_bacc, final_acc, final_f1, final_precision, final_recall,precisions


def adapt(args):
    fix_random_seed_as(args.seed)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not args.output_dir:
        args.output_dir = datetime.now().strftime("%Y%m%d%H%M%S")
    export_root = os.path.join(EXPERIMENT_ROOT_FOLDER, args.output_dir)
    if not os.path.exists(export_root):
        os.makedirs(export_root)
    
    tokenizer = AutoTokenizer.from_pretrained(export_root)
    # test data
    test_dataloader = get_target_loader(args, mode='target_test', tokenizer=tokenizer)
    
    model = AutoModelForSequenceClassification.from_pretrained(export_root)
    model = model.to(args.device)


    best_bacc, best_acc, best_f1, precision, recall,precisions= evaluation(args, model, test_dataloader)
    
    result = [best_bacc, best_acc, best_f1,precision,recall]
    # only save result file
    print({"pos_pre":precisions[1],"neg_pre":precisions[0]})
    with open(os.path.join(export_root, 'eval_test_metrics.json'), 'w') as f:
        json.dump(result, f) 
 

if __name__ == '__main__':
    print(args)
    adapt(args)
