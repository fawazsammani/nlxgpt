import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import GPT2Tokenizer, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup
import json
from cococaption.pycocotools.coco import COCO
from cococaption.pycocoevalcap.eval import COCOEvalCap
from PIL import Image
from accelerate import Accelerator
from models.gpt import GPT2LMHeadModel
from models.clip_vit import ImageEncoder
from utils.eval_utils import top_filtering
from utils.data_utils import *


def change_requires_grad(model, req_grad):
    for p in model.parameters():
        p.requires_grad = req_grad
        
def load_pretrained():
    
    model_path = 'pretrained_model/pretrain_model_14'
    tokenizer_path = 'pretrained_model/pretrain_tokenizer_0'
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)        # load tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)   # load model with config
    return tokenizer, model


def load_checkpoint(ckpt_path, epoch):
    
    model_name = 'nle_model_{}'.format(str(epoch))
    tokenizer_name = 'nle_gpt2_tokenizer_0'
    filename = 'ckpt_stats_' + str(epoch) + '.tar'
    
    tokenizer = GPT2Tokenizer.from_pretrained(ckpt_path + tokenizer_name)        # load tokenizer
    model = GPT2LMHeadModel.from_pretrained(ckpt_path + model_name).to(device)   # load model with config
    opt = torch.load(ckpt_path + filename)
    optimizer = get_optimizer(model, learning_rate)
    optimizer.load_state_dict(opt['optimizer_state_dict'])
    start_epoch = opt['epoch'] + 1
    scheduler_dic = opt['scheduler']
    del opt
    torch.cuda.empty_cache()

    return tokenizer, model, optimizer, scheduler_dic, start_epoch
    

def save_checkpoint(epoch, unwrapped_model, optimizer, tokenizer, scheduler, ckpt_path, **kwargs):
    
    model_name = 'nle_model_{}'.format(str(epoch))
    tokenizer_name = 'nle_gpt2_tokenizer_{}'.format(str(epoch))
    filename = 'ckpt_stats_' + str(epoch) + '.tar'
    
    if epoch == 0:
        tokenizer.save_pretrained(ckpt_path + tokenizer_name)   # save tokenizer
        
    unwrapped_model.save_pretrained(ckpt_path + model_name, save_function=accelerator.save)
        
    opt = {'epoch': epoch,
           'optimizer_state_dict': optimizer.state_dict(), 
           'scheduler': scheduler.state_dict(),
            **kwargs}
    
    accelerator.save(opt, ckpt_path + filename)
    
# def get_scores(annFile, resFile, save_scores_path):
    
#     coco = COCO(annFile)
#     cocoRes = coco.loadRes(resFile)
#     cocoEval = COCOEvalCap(coco, cocoRes)
#     cocoEval.evaluate()
#     with open(save_scores_path, 'w') as w:
#         json.dump(cocoEval.eval, w)
    
def filter_and_get_scores(resFileExp, save_scores_pathExp, full_predictions, exp_predictions):

    annotFull = json.load(open(annFileFull, 'r'))
    
    gt_answers = {}
    for item in annotFull['annotations']:
        gt_answers[item['image_id']] = item['caption'].split("because")[0].strip()
        
    pred_answers = {}
    for item in full_predictions:
        pred_answers[item['image_id']] = item['caption'].split("because")[0].strip()
        
    correct_keys = []
    for key,value in pred_answers.items():
        gt_answer = gt_answers[key]
        if value == gt_answer:
            correct_keys.append(key)
            
    exp_preds = [item for item in exp_predictions if item['image_id'] in correct_keys]
    
    with open(resFileExp, 'w') as w:
        json.dump(exp_preds, w)
        
        
    coco = COCO(annFileExp)
    cocoRes = coco.loadRes(resFileExp)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()
    
    with open(save_scores_pathExp, 'w') as w:
        json.dump(cocoEval.eval, w)
    

class ACTXTrainDataset(Dataset):

    def __init__(self, path, transform, tokenizer, max_seq_len):
        
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len       # <bos> The answer is <answer> becase <explanation> <eos>
        self.data = json.load(open(path, 'r'))
        self.ids_list = list(self.data.keys())
        
        for k,v in self.data.items():   
            if len(v['explanation']) > 1:   # take care of questions that have more than one explanation
                # duplicate them for loading. -1 because one explanation is already in ids_list
                self.ids_list += [str(k)] * (len(v['explanation']) - 1)    

        self.index_tracker = {k: len(v['explanation']) - 1 for k,v in self.data.items()}
        

    def __getitem__(self, i):
        
        img_id = self.ids_list[i]
        sample = self.data[img_id]
        img_name = sample['image_name']
        answer = data_utils.prep_ans(sample['answers'])

        exp_idx = self.index_tracker[img_id]   # the index of the explanation for questions with multiple explanations
        if exp_idx > 0:
            self.index_tracker[img_id] -= 1    # decrease usage
                
        text_exp = sample['explanation'][exp_idx]     # explanation
        
        # tokenization process
        a_segment_id, e_segment_id = self.tokenizer.convert_tokens_to_ids(['<answer>', '<explanation>'])

        tokens_a = [self.tokenizer.bos_token] + self.tokenizer.tokenize("the answer is " + answer)
        answer_len = len(tokens_a)
        segment_ids = [a_segment_id] * answer_len
        
        tokens_b = self.tokenizer.tokenize(" because " + text_exp) + [self.tokenizer.eos_token]
        exp_len = len(tokens_b)
        segment_ids += [e_segment_id] * exp_len
        
        tokens = tokens_a + tokens_b
        labels = [-100] + tokens_a[1:] + tokens_b   # labels will be shifted in the model, so for now set them same as tokens
        
        if len(tokens) > self.max_seq_len :
            tokens = tokens[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
            segment_ids = segment_ids[:self.max_seq_len]

        assert len(tokens) == len(segment_ids) 
        assert len(tokens) == len(labels)
        
        seq_len = len(tokens)
        padding_len = self.max_seq_len - seq_len
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)
        labels = labels + ([-100] * padding_len)
        segment_ids += ([e_segment_id] * padding_len)
        
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        labels = [self.tokenizer.convert_tokens_to_ids(t) if t!=-100 else t for t in labels]
        labels = torch.tensor(labels, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        
        folder = 'images/mpi/'
        img_path = folder + img_name
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img_id = torch.LongTensor([int(img_id)])
        
        return (img, img_id, input_ids, labels, segment_ids)

    def __len__(self):
        return len(self.ids_list)
    
class ACTXEvalDataset(Dataset):

    def __init__(self, path, transform, tokenizer, max_seq_len):

        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len       # question + <bos> The answer is <answer> becase <explanation> <eos>
        self.data = json.load(open(path, 'r'))
        self.ids_list = list(self.data.keys())


    def __getitem__(self, i):
        
        img_id = self.ids_list[i]
        sample = self.data[img_id]
        img_name = sample['image_name']

        # tokenization process
        a_segment_id, e_segment_id = self.tokenizer.convert_tokens_to_ids(['<answer>', '<explanation>'])

        tokens = [self.tokenizer.bos_token] + self.tokenizer.tokenize("the answer is")
        answer_len = len(tokens)
        segment_ids = [a_segment_id] * answer_len

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        
        folder = 'images/mpi/'
        img_path = folder + img_name
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img_id = torch.LongTensor([int(img_id)])
        
        return (img, img_id, input_ids, segment_ids)

    def __len__(self):
        return len(self.ids_list)


def sample_sequences(model, tokenizer, loader):
    
    model.eval()
    results_exp = []
    results_full = []
    SPECIAL_TOKENS = ['<|endoftext|>', '<pad>', '<question>', '<answer>', '<explanation>']
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    because_token = tokenizer.convert_tokens_to_ids('Ġbecause')
    max_len = 20
    
    for i,batch in enumerate(loader):
        
        current_output = []
        batch = tuple(input_tensor.to(device) for input_tensor in batch)
        img, img_id, input_ids, segment_ids = batch
        img_embeddings = image_encoder(img)
        always_exp = False
        
        with torch.no_grad():
            
            for step in range(max_len + 1):
                
                if step == max_len:
                    break
                
                outputs = model(input_ids=input_ids, 
                                past_key_values=None, 
                                attention_mask=None, 
                                token_type_ids=segment_ids, 
                                position_ids=None, 
                                encoder_hidden_states=img_embeddings, 
                                encoder_attention_mask=None, 
                                labels=None, 
                                use_cache=False, 
                                return_dict=True)
                
                lm_logits = outputs.logits 
                logits = lm_logits[0, -1, :] / temperature
                logits = top_filtering(logits, top_k=top_k, top_p=top_p)
                probs = F.softmax(logits, dim=-1)
                prev = torch.topk(probs, 1)[1] if no_sample else torch.multinomial(probs, 1)
                
                if prev.item() in special_tokens_ids:
                    break
                
                # take care of when to start the <explanation> token
                if not always_exp:
                    
                    if prev.item() != because_token:
                        new_segment = special_tokens_ids[-2]   # answer segment
                    else:
                        new_segment = special_tokens_ids[-1]   # explanation segment
                        always_exp = True
                else:
                    new_segment = special_tokens_ids[-1]   # explanation segment
                    
                new_segment = torch.LongTensor([new_segment]).to(device)
                current_output.append(prev.item())
                input_ids = torch.cat((input_ids, prev.unsqueeze(0)), dim = 1)
                segment_ids = torch.cat((segment_ids, new_segment.unsqueeze(0)), dim = 1)
                
        decoded_sequences = tokenizer.decode(current_output, skip_special_tokens=True).lstrip()
        results_full.append({"image_id": img_id.item(), "caption": decoded_sequences})
        
        if 'because' in decoded_sequences:
            cut_decoded_sequences = decoded_sequences.split('because')[-1].strip()
        else:
            cut_decoded_sequences = " ".join(decoded_sequences.split()[2:])
        
        results_exp.append({"image_id": img_id.item(), "caption": cut_decoded_sequences})
        print("\rEvaluation: Finished {}/{}".format(i, len(loader)), end='          ')
            
    return results_full, results_exp

def get_optimizer(model, learning_rate):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],  
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer


accelerator = Accelerator()
device = accelerator.device

finetune_pretrained = False   # if True, finetunes from the image captioning model
eval_batch_size = 1
img_size = 224
ckpt_path = 'ckpts/'
caption_save_path = 'cococaption/results/' 
annFileExp = 'cococaption/annotations/actX_test_annot_exp.json'
annFileFull = 'cococaption/annotations/actX_test_annot_full.json'
max_seq_len = 30
load_from_epoch = None
no_sample = True  
top_k =  0
top_p =  0.9
batch_size = 32   # per GPU
num_train_epochs = 30
weight_decay = 0
learning_rate = 2e-5 if not finetune_pretrained else 1e-5
gradient_accumulation_steps = 1   
start_epoch = 0
temperature = 1

image_encoder = ImageEncoder(device).to(device)
change_requires_grad(image_encoder, False)


if load_from_epoch is not None:
    tokenizer, model, optimizer, scheduler_dic, start_epoch = load_checkpoint(ckpt_path, load_from_epoch)
    
else:
    
    if finetune_pretrained:
        tokenizer, model = load_pretrained()
        optimizer = get_optimizer(model, learning_rate)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        orig_num_tokens = len(tokenizer.encoder)
        
        num_new_tokens = tokenizer.add_special_tokens({'pad_token': '<pad>',
                                                       'additional_special_tokens': ['<answer>', '<explanation>']})
        
        assert len(tokenizer) == orig_num_tokens + num_new_tokens
        
        config = AutoConfig.from_pretrained('distilgpt2')
        
        # Add configs
        setattr(config, 'img_size', None)
        setattr(config, 'max_seq_len', None)   
        config.img_size = img_size
        config.max_seq_len = max_seq_len 
        config.add_cross_attention = True
        
        model = GPT2LMHeadModel.from_pretrained('distilgpt2', config = config)
        model.resize_token_embeddings(len(tokenizer))
        model = model.to(device)
        optimizer = get_optimizer(model, learning_rate)
        
print("Model Setup Ready...")


img_transform = transforms.Compose([transforms.Resize((img_size,img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_dataset = ACTXTrainDataset(path = 'nle_data/ACT-X/actX_train.json', 
                                 transform = img_transform, 
                                 tokenizer = tokenizer, 
                                 max_seq_len = max_seq_len)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size = batch_size, 
                                           shuffle=True, 
                                           pin_memory=True)

# val_dataset = ACTXEvalDataset(path = 'nle_data/ACT-X/actX_val.json',      
#                               transform = img_transform, 
#                               tokenizer = tokenizer, 
#                               max_seq_len = max_seq_len)


# val_loader = torch.utils.data.DataLoader(val_dataset,
#                                          batch_size = eval_batch_size, 
#                                          shuffle=False, 
#                                          pin_memory=True)

test_dataset = ACTXEvalDataset(path = 'nle_data/ACT-X/actX_test.json', 
                               transform = img_transform, 
                               tokenizer = tokenizer, 
                               max_seq_len = max_seq_len)


test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size = eval_batch_size, 
                                          shuffle=False, 
                                          pin_memory=True)


model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

t_total = (len(train_loader) // gradient_accumulation_steps) * num_train_epochs
warmup_steps = 0   # 0.10 * t_total
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

if load_from_epoch is not None:
    scheduler.load_state_dict(scheduler_dic)


for epoch in range(start_epoch, num_train_epochs):
    
    model.train()
    accum_loss = 0
    
    for step, batch in enumerate(train_loader):
        
        batch = tuple(input_tensor.to(device) for input_tensor in batch)
        img, _, input_ids, labels, segment_ids = batch
        
        img_embeddings = image_encoder(img)
        
        outputs = model(input_ids=input_ids, 
                        past_key_values=None, 
                        attention_mask=None, 
                        token_type_ids=segment_ids, 
                        position_ids=None, 
                        encoder_hidden_states=img_embeddings, 
                        encoder_attention_mask=None, 
                        labels=labels, 
                        use_cache=False, 
                        return_dict=True)
        
        loss = outputs.loss
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)
        accum_loss += loss.item()
        
        if step % gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            accelerator.print("\rEpoch {} / {}, Iter {} / {}, Loss: {:.3f}".format(epoch, 
                                                                                   num_train_epochs, 
                                                                                   step, len(train_loader), 
                                                                                   accum_loss), 
                              end='          ')
            accum_loss = 0
            
            
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    save_checkpoint(epoch, unwrapped_model, optimizer, tokenizer, scheduler, ckpt_path)
                                                                                     
    if accelerator.is_main_process:     
        
        results_full, results_exp = sample_sequences(unwrapped_model, tokenizer, test_loader)
        
        resFileExp = caption_save_path + 'captions_exp_' + str(epoch) + '.json'
        unf_resFileExp = caption_save_path + 'unf_captions_exp_' + str(epoch) + '.json'
        unf_resFileFull = caption_save_path + 'unf_captions_full_' + str(epoch) + '.json'
        save_scores_pathExp = caption_save_path + 'scores_exp_' + str(epoch) + '.json'
        
        with open(unf_resFileExp, 'w') as w:
            json.dump(results_exp, w)
            
        with open(unf_resFileFull, 'w') as w:
            json.dump(results_full, w)
        
        # unfiltered results
        # get_scores(annFileExp, unf_resFileExp, save_scores_pathExp)
        
        # filtered results
        filter_and_get_scores(resFileExp, save_scores_pathExp, results_full, results_exp)
    
