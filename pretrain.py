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


def change_requires_grad(model, req_grad):
    for p in model.parameters():
        p.requires_grad = req_grad


def load_checkpoint(ckpt_path, epoch):
    
    model_name = 'pretrain_model_{}'.format(str(epoch))
    tokenizer_name = 'pretrain_tokenizer_0'
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
    
    model_name = 'pretrain_model_{}'.format(str(epoch))
    tokenizer_name = 'pretrain_tokenizer_{}'.format(str(epoch))
    filename = 'ckpt_stats_' + str(epoch) + '.tar'
    
    if epoch == 0:
        tokenizer.save_pretrained(ckpt_path + tokenizer_name)   # save tokenizer
        
    unwrapped_model.save_pretrained(ckpt_path + model_name, save_function=accelerator.save)
        
    opt = {'epoch': epoch,
           'optimizer_state_dict': optimizer.state_dict(), 
           'scheduler': scheduler.state_dict(),
            **kwargs}
    
    accelerator.save(opt, ckpt_path + filename)
    
def get_scores(annFile, resFile, save_scores_path):
    
    coco = COCO(annFile)
    cocoRes = coco.loadRes(resFile)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()
    with open(save_scores_path, 'w') as w:
        json.dump(cocoEval.eval, w)
        
    return cocoEval.eval['CIDEr']

class PretrainCaptioning(Dataset):

    def __init__(self, 
                 all_imgs_folder,
                 imgs_path, 
                 captions_path,
                 split, 
                 transform, 
                 tokenizer, 
                 max_seq_len):
        
        self.split = split
        self.transform = transform
        self.all_imgs_folder = all_imgs_folder
        self.images_list = json.load(open(imgs_path, 'r'))
        
        if split == 'train':
            
            self.max_seq_len = max_seq_len   # caption (with <bos>, <eos> and padding)
            self.tokenizer = tokenizer
            self.captions = json.load(open(captions_path, 'r'))
                

    def __getitem__(self, i):
        
        img_subpath =  self.images_list[i]
        img_path = self.all_imgs_folder + img_subpath
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        
        if self.split == 'train':
            labels = []
            text = self.captions[i]
            tokens = self.tokenizer.tokenize(text)
            tokens =  [self.tokenizer.bos_token] + tokens + [self.tokenizer.eos_token]
            labels += tokens

            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
                labels = labels[:self.max_seq_len]

            seq_len = len(tokens)
            padding_len = self.max_seq_len - seq_len
            tokens += [self.tokenizer.pad_token] * padding_len
            labels += ([-100] * padding_len)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            
            labels = [self.tokenizer.convert_tokens_to_ids(t) if t!=-100 else t for t in labels]
            labels = torch.tensor(labels, dtype=torch.long)

        if self.split != 'train':
            img_id = int(img_subpath.split("_")[-1].split(".")[0].lstrip("0"))
            img_id = torch.LongTensor([img_id])
            return (img, img_id)
        
        return (img, input_ids, labels)
        
    def __len__(self):
        return len(self.images_list)


def sample_sequences(model, tokenizer, loader):
    
    model.eval()
    results = []
    SPECIAL_TOKENS = ['<|endoftext|>', '<pad>']
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    
    for i,batch in enumerate(loader):
        
        current_output = []
        batch = tuple(input_tensor.to(device) for input_tensor in batch)
        img, img_id = batch
        img_embeddings = image_encoder(img)
        max_len = 20
        batch_size = img_embeddings.size(0)
        input_ids = torch.LongTensor(batch_size).to(device)  
        input_ids[:] = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
        input_ids = input_ids.unsqueeze(1)   # (batch_size, 1) 
        
        with torch.no_grad():
            
            for step in range(max_len + 1):
                
                if step == max_len:
                    break
                
                outputs = model(input_ids=input_ids, 
                                past_key_values=None, 
                                attention_mask=None, 
                                token_type_ids=None, 
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

                current_output.append(prev.item())
                input_ids = torch.cat((input_ids, prev.unsqueeze(0)), dim = 1)
                
        decoded_sequences = tokenizer.decode(current_output, skip_special_tokens=True).lstrip()
        results.append({"image_id": img_id.item(), "caption": decoded_sequences})
        print("\rEvaluation: Finished {}/{}".format(i, len(loader)), end='          ')
            
    return results

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

eval_batch_size = 1
img_size = 224
ckpt_path = 'ckpts/'
caption_save_path = 'cococaption/results/' 
annFile = 'cococaption/annotations/captions_val2014.json' 
max_seq_len = 70
load_from_epoch = None
no_sample = True   
top_k =  0
top_p =  0.9
batch_size = 32   # per GPU. Total batch size: 576
num_train_epochs = 30
weight_decay = 0 
learning_rate = 1e-4  
gradient_accumulation_steps = 6    # accum_steps = desired_batch_size per GPU / tolerable_batch_size per GPU
start_epoch = 0
temperature = 1

image_encoder = ImageEncoder(device).to(device)
change_requires_grad(image_encoder, False)


if load_from_epoch is None:

    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    orig_num_tokens = len(tokenizer.encoder)
    # add the additional tokens here to avoid changing the tokenizer and model weights for every downstream task
    num_new_tokens = tokenizer.add_special_tokens({'pad_token': '<pad>',
                                                   'additional_special_tokens': ['<question>', '<answer>', '<explanation>']})
    
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
    
else:
    tokenizer, model, optimizer, scheduler_dic, start_epoch = load_checkpoint(ckpt_path, load_from_epoch)


img_transform = transforms.Compose([transforms.Resize((img_size,img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_dataset = PretrainCaptioning(all_imgs_folder = 'images/',
                                   imgs_path = 'pretrain_data/corpus_images_train.json', 
                                   captions_path = 'pretrain_data/corpus_captions_train.json', 
                                   split = 'train', 
                                   transform = img_transform, 
                                   tokenizer = tokenizer, 
                                   max_seq_len = max_seq_len)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size = batch_size, 
                                           shuffle=True, 
                                           pin_memory=True)

test_dataset = PretrainCaptioning(all_imgs_folder = 'images/',
                                  imgs_path = 'pretrain_data/corpus_images_test.json', 
                                  captions_path = None, 
                                  split = 'test', 
                                  transform = img_transform, 
                                  tokenizer = None, 
                                  max_seq_len = None)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size = eval_batch_size, 
                                          shuffle=False, 
                                          pin_memory=True)


model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

t_total = (len(train_loader) // gradient_accumulation_steps) * num_train_epochs
warmup_steps = 0    # int(0.10 * t_total)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

if load_from_epoch is not None:
    scheduler.load_state_dict(scheduler_dic)


for epoch in range(start_epoch, num_train_epochs):
    
    model.train()
    accum_loss = 0
    
    for step, batch in enumerate(train_loader):
        
        batch = tuple(input_tensor.to(device) for input_tensor in batch)
        img, input_ids, labels = batch
        
        img_embeddings = image_encoder(img)
        
        outputs = model(input_ids=input_ids, 
                        past_key_values=None, 
                        attention_mask=None, 
                        token_type_ids=None, 
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
        
        results = sample_sequences(unwrapped_model, tokenizer, test_loader)
        
        resFile = caption_save_path + 'captions_' + str(epoch) + '.json'
        save_scores_path = caption_save_path + 'scores_' + str(epoch) + '.json'
        
        with open(resFile, 'w') as w:
            json.dump(results, w)
            
        get_scores(annFile, resFile, save_scores_path)
    