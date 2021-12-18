import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, AutoConfig, DistilBertPreTrainedModel, DistilBertModel 
from torch.nn import BCEWithLogitsLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AdamW, get_linear_schedule_with_warmup
import json
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

def get_score(occur):
    if occur == 0:
        return .0
    elif occur == 1:
        return .3
    elif occur == 2:
        return .6
    elif occur == 3:
        return .9
    else:
        return 1.
    
    
def get_answer_labels(ans, ans_to_ix):
    ans_score = np.zeros(ans_to_ix.__len__(), np.float32)
    ans_prob_dict = {}

    for ans_proc in ans:
        if ans_proc not in ans_prob_dict:
            ans_prob_dict[ans_proc] = 1
        else:
            ans_prob_dict[ans_proc] += 1

    for ans_ in ans_prob_dict:
        if ans_ in ans_to_ix:
            ans_score[ans_to_ix[ans_]] = get_score(ans_prob_dict[ans_])

    return ans_score



def load_checkpoint(ckpt_path, epoch):
    
    model_name = 'ep_distilbert_model_{}'.format(str(epoch))
    tokenizer_name = 'ep_distilbert_tokenizer_0'
    filename = 'ckpt_stats_' + str(epoch) + '.tar'
    
    tokenizer = DistilBertTokenizer.from_pretrained(ckpt_path + tokenizer_name)        # load tokenizer
    model = DistilBertForSequenceClassification.from_pretrained(ckpt_path + model_name).to(device)   # load model with config
    opt = torch.load(ckpt_path + filename)
    optimizer = get_optimizer(model, learning_rate)
    optimizer.load_state_dict(opt['optimizer_state_dict'])
    start_epoch = opt['epoch'] + 1
    scheduler_dic = opt['scheduler']
    del opt
    torch.cuda.empty_cache()

    return tokenizer, model, optimizer, scheduler_dic, start_epoch
    

def save_checkpoint(epoch, model, optimizer, tokenizer, scheduler, ckpt_path, **kwargs):
    
    model_name = 'ep_distilbert_model_{}'.format(str(epoch))
    tokenizer_name = 'ep_distilbert_tokenizer_0'
    filename = 'ckpt_stats_' + str(epoch) + '.tar'
    
    if epoch == 0:
        tokenizer.save_pretrained(ckpt_path + tokenizer_name)   # save tokenizer
        
    model.save_pretrained(ckpt_path + model_name)
        
    opt = {'epoch': epoch,
           'optimizer_state_dict': optimizer.state_dict(), 
           'scheduler': scheduler.state_dict(),
            **kwargs}
    
    torch.save(opt, ckpt_path + filename)
    

class ExplainPredictDataset(Dataset):

    def __init__(self, data_path, answer_path, tokenizer, max_seq_len, gt_test):
        
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len     
        self.data = json.load(open(data_path, 'r'))
        self.answer_dic = json.load(open(answer_path, 'r'))
        self.ids_list = list(self.data.keys())
        self.train_mode = 'train' in data_path
        self.gt_test = gt_test
        
    def __getitem__(self, index):
        
        quention_id = self.ids_list[index]
        qid = torch.LongTensor([int(quention_id)])
        sample = self.data[quention_id]
        text_a = sample['question']
        
        if self.train_mode:
            text_b = sample['explanation']
        else:
            text_b = sample['gt_explanation'] if self.gt_test  else sample['explanation']
            
        answer = get_answer_labels(sample['all_answers_raw'], self.answer_dic)
        answer = torch.from_numpy(answer)

        # tokenization process
        tokens = [self.tokenizer.cls_token] + self.tokenizer.tokenize(text_a) 
        
        if len(tokens) > 15 :
            tokens = tokens[:15]

        tokens_b = [self.tokenizer.sep_token] + self.tokenizer.tokenize(text_b) 
        tokens += tokens_b

        seq_len = len(tokens)
        
        if seq_len > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        
        padding_len = self.max_seq_len - seq_len
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_masks = input_ids != tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

        return (input_ids, attention_masks, answer, qid)

    def __len__(self):
        return len(self.ids_list)
    
class DistilBertForSequenceClassification(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss = None
        if labels is not None:
            loss_fct = BCEWithLogitsLoss(reduction='sum')
            loss = loss_fct(logits, labels)
                
        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )

    

def evaluate(model, inv_answer_dic, loader):
    
    model.eval()
    correct_ids = []

    print("Evaluating....................")
    
    for i, (input_ids, attention_masks, _, qid) in enumerate(loader):
        
        input_ids = input_ids.to(device)  
        attention_masks = attention_masks.to(device) 
        
        with torch.no_grad():
            predictions = model(input_ids=input_ids, 
                                attention_mask=attention_masks, 
                                head_mask=None, 
                                inputs_embeds=None,
                                labels=None,
                                output_attentions=False,
                                output_hidden_states=False,
                                return_dict=True)
            
        _, ind = torch.max(predictions.logits, dim=-1)

        for i,pred in enumerate(ind):
            pred_answer = inv_answer_dic[pred.item()]
            gt_answers = list(set(tst_data[str(qid[i].item())]['all_answers_raw']))
            if pred_answer in gt_answers:
                correct_ids.append(qid[i].item())
        
    accuracy = len(correct_ids) / len(test_loader.dataset)
    print("Accuracy: {:.2f} %".format(accuracy * 100))

    return accuracy, correct_ids
        

def get_optimizer(model, learning_rate):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],  
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer


ckpt_path = 'ckpts/'
answer_dic_file = 'data/answer_dic.json'
train_data_file = 'data/train_data.json'
test_data_file = 'data/test_data.json'
max_seq_len = 40
load_from_epoch = None
batch_size = 16  
num_train_epochs = 15
weight_decay = 0
learning_rate = 2e-5
start_epoch = 0
best_accuracy = 0

answer_dic = json.load(open(answer_dic_file, 'r'))
inv_answer_dic = {v: k for k, v in answer_dic.items()}
tst_data = json.load(open(test_data_file, 'r'))


if load_from_epoch is None:

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    config = AutoConfig.from_pretrained('distilbert-base-uncased')
    config.num_labels = len(answer_dic)

    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', config = config)
    model = model.to(device)
    optimizer = get_optimizer(model, learning_rate)
    
else:
    tokenizer, model, optimizer, scheduler_dic, start_epoch = load_checkpoint(ckpt_path, load_from_epoch)



train_dataset = ExplainPredictDataset(data_path = train_data_file, 
                                      answer_path = answer_dic_file, 
                                      tokenizer = tokenizer, 
                                      max_seq_len = max_seq_len, 
                                      gt_test = None)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size = batch_size, 
                                           shuffle=True, 
                                           pin_memory=True)

test_dataset = ExplainPredictDataset(data_path = test_data_file, 
                                     answer_path = answer_dic_file, 
                                     tokenizer = tokenizer, 
                                     max_seq_len = max_seq_len, 
                                     gt_test = True)


test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size = batch_size, 
                                          shuffle=False, 
                                          pin_memory=True)


t_total = len(train_loader) * num_train_epochs
warmup_steps = 0
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

if load_from_epoch is not None:
    scheduler.load_state_dict(scheduler_dic)


for epoch in range(start_epoch, num_train_epochs):
    
    model.train()
    accum_loss = 0
    
    for step, batch in enumerate(train_loader):
        
        
        batch = tuple(input_tensor.to(device) for input_tensor in batch)
        input_ids, attention_masks, answer, _ = batch
        
        outputs = model(input_ids=input_ids, 
                        attention_mask=attention_masks, 
                        head_mask=None, 
                        inputs_embeds=None,
                        labels=answer,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True)
        
        loss = outputs.loss
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        print("\rEpoch {} / {}, Iter {} / {}, Loss: {:.3f}".format(epoch, 
                                                                   num_train_epochs, 
                                                                   step, len(train_loader), 
                                                                   loss.item()), 
              end='          ')
            
            
    
    accuracy, _ = evaluate(model, inv_answer_dic, test_loader)
    
    if accuracy > best_accuracy:
        save_checkpoint(epoch, model, optimizer, tokenizer, scheduler, ckpt_path)
        best_accuracy = accuracy * 1.0
        best_epoch = epoch * 1
        
print("----------------------------------------")
print("Best Accuracy is: {:.2f}".format(best_accuracy * 100))
print("----------------------------------------")

# Evaluate on the GPT Data
tokenizer, model, *_ = load_checkpoint(ckpt_path, best_epoch)
print("Loaded from ", best_epoch)

test_dataset = ExplainPredictDataset(data_path = test_data_file, 
                                     answer_path = answer_dic_file, 
                                     tokenizer = tokenizer, 
                                     max_seq_len = max_seq_len, 
                                     gt_test = False)


test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size = batch_size, 
                                          shuffle=False, 
                                          pin_memory=True)


accuracy, correct_ids = evaluate(model, inv_answer_dic, test_loader)

print("----------------------------------------")
print("Model EP Accuracy is: {:.2f}".format(accuracy * 100))
print("----------------------------------------")
        
    