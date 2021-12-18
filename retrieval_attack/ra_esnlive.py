import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoConfig
from sentence_transformers import SentenceTransformer
import json
import clip
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)


def change_requires_grad(model, req_grad):
    for p in model.parameters():
        p.requires_grad = req_grad


def load_checkpoint(ckpt_path, epoch):
    
    model_name = 'nle_model_{}'.format(str(epoch))
    tokenizer_name = 'nle_gpt2_tokenizer_0'
    tokenizer = GPT2Tokenizer.from_pretrained(ckpt_path + tokenizer_name)        # load tokenizer
    model = GPT2LMHeadModel.from_pretrained(ckpt_path + model_name).to(device)   # load model with config

    return tokenizer, model

def get_batch_concept_ids(predictions):

    _, ind = predictions.topk(20, 1, True, True)
    concept_ids = []
    for i in range(ind.size(0)):
        p_concepts = [rev_wordmap[a.item()] for a in ind[i]]
        p_concepts = [c if "_" not in c else " ".join(c.split("_")) for c in p_concepts]
        tokenized = tokenizer.tokenize(" " + " ".join(p_concepts))
        token_ids = tokenizer.convert_tokens_to_ids(tokenized)[:20]
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        concept_ids.append(token_ids.unsqueeze(0))

    concept_ids = torch.cat(concept_ids, dim=0)
    return concept_ids.to(device)

class AttFlat(nn.Module):

    def __init__(self, dim, drop):
        super(AttFlat, self).__init__()

        self.mlp = nn.Sequential(nn.Linear(dim, dim), 
                                 nn.ReLU(), 
                                 nn.Dropout(drop), 
                                 nn.Linear(dim, 1))
        
    def forward(self, x):
        
        att = self.mlp(x)   
        att = att.squeeze(-1) 
        att = F.softmax(att, dim=-1)
        x_atted = (x * att.unsqueeze(-1)).sum(dim=1)
        return x_atted
    
    
class LinearProbe(nn.Module):

    def __init__(self, dim, drop, vocab_size):
        super().__init__()
        
        self.linear = nn.Sequential(nn.Linear(dim, dim), 
                                    nn.ReLU(), 
                                    nn.Dropout(drop), 
                                    nn.Linear(dim, dim))
        
        self.layer_norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop)
        self.att_flat = AttFlat(dim, drop)
        self.classifier = nn.Linear(dim, vocab_size)
        
    def forward(self, x):
        x = x + self.drop(self.linear(x))
        x = self.layer_norm(x)
        x = self.att_flat(x)
        x = self.classifier(x)
        return x
    

class ImageEncoder(nn.Module):

    def __init__(self):
        super(ImageEncoder, self).__init__()

        self.encoder, _ = clip.load("ViT-B/16", device=device)   # loads already in eval mode

    def forward(self, x):
        """
        Expects a tensor of size (batch_size, 3, 224, 224)
        """
        with torch.no_grad():
            x = x.type(self.encoder.visual.conv1.weight.dtype)
            x = self.encoder.visual.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat([self.encoder.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            x = x + self.encoder.visual.positional_embedding.to(x.dtype)
            x = self.encoder.visual.ln_pre(x)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.encoder.visual.transformer(x)
            grid_feats = x.permute(1, 0, 2)  # LND -> NLD    (N, 197, 768)
            grid_feats = self.encoder.visual.ln_post(grid_feats[:,1:])  

        return grid_feats.float()
    

def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):

    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequences(img, model, input_ids, tokenizer):
    
    SPECIAL_TOKENS = ['<|endoftext|>', '<pad>']
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    max_len = 20
    current_output = []
    img_embeddings = image_encoder(img)
        
    with torch.no_grad():
        
        predictions = torch.sigmoid(linear_probe(img_embeddings))
        concept_ids = get_batch_concept_ids(predictions)
        input_ids = torch.cat([concept_ids, input_ids], dim = 1)
        
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

        if 'because' in decoded_sequences:
            cut_decoded_sequences = decoded_sequences.split('because')[-1].strip()
        else:
            cut_decoded_sequences = " ".join(decoded_sequences.split()[2:])

    return cut_decoded_sequences


eval_batch_size = 1
img_size = 224
ckpt_path = 'ckpts/'
nle_data_test_path = 'nle_data/eSNLI-VE/esnlive_test.json'
img_folder = 'images/flickr30k/'
max_seq_len = 40
load_from_epoch = 4
no_sample = True  
top_k =  0
top_p =  0.9
temperature = 1

image_encoder = ImageEncoder().to(device)
change_requires_grad(image_encoder, False)
tokenizer, model = load_checkpoint(ckpt_path, load_from_epoch)
model.eval()

# load sentence-bert encoder
sbert_model = SentenceTransformer('all-mpnet-base-v2').to(device)
#sbert_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
sbert_model.eval()

wordmap = json.load(open('nle_data/eSNLI-VE/vg_concept_word2index.json', 'r'))
rev_wordmap = {v: k for k, v in wordmap.items()}

linear_probe = LinearProbe(768, 0.1, len(wordmap)).to(device)
linear_probe.load_state_dict(torch.load('pretrained_model/linear_probe_12.tar')['model'])
change_requires_grad(linear_probe, False)

img_transform = transforms.Compose([transforms.Resize((img_size,img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def get_inputs(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    answer = [tokenizer.bos_token] + tokenizer.tokenize(" the answer is")
    tokens += answer 
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    return input_ids.unsqueeze(0).to(device)

def get_distances(matrix):
    matrix /= matrix.norm(dim=-1, keepdim=True)
    sim = torch.mm(matrix, matrix.T).clamp(min=0)  
    triu_mask = (torch.triu(torch.ones_like(sim), diagonal=1)).to(device)
    dist = ((sim * triu_mask).sum()) / (triu_mask.sum().float())
    return dist.item()


# load retrieval attack data
i2t = json.load(open('ra/retrieval_attack_data_i2t_esnlive.json', 'r'))
t2i = json.load(open('ra/retrieval_attack_data_t2i_esnlive.json', 'r'))

# i2t
distances_5 = []
distances_10 = []
distances_15 = []

for i,(img_name, questions) in enumerate(i2t.items()):
    
    all_vectors = []
    
    img_path = img_folder + img_name
    img = Image.open(img_path).convert('RGB')
    img = img_transform(img).unsqueeze(0).to(device)
    p_questions = questions[:15]
    
    for text in p_questions:
        input_ids = get_inputs(text, tokenizer)
        seq = sample_sequences(img, model, input_ids, tokenizer)
        encoding = sbert_model.encode([seq], convert_to_tensor=True)
        all_vectors.append(encoding)
        
    all_vectors = torch.cat(all_vectors, dim=0)     # (15, 768)
    
    distances_5.append(get_distances(all_vectors[:5, :]))
    distances_10.append(get_distances(all_vectors[:10, :]))
    distances_15.append(get_distances(all_vectors))
    
    print("\rFinished {}/{} images".format(i+1, len(i2t)), end='          ')
    
mean5_dist = sum(distances_5) / len(distances_5)
mean10_dist = sum(distances_10) / len(distances_10)
mean15_dist = sum(distances_15) / len(distances_15)

print("RA Image2Text @5: {}".format(mean5_dist ))
print("RA Image2Text @10: {}".format(mean10_dist))
print("RA Image2Text @15: {}".format(mean15_dist))
        
    
# t2i
distances_5 = []
distances_10 = []
distances_15 = []

for i, (text, img_names) in enumerate(t2i.items()):
    
    all_vectors = []
    input_ids = get_inputs(text, tokenizer)
    p_img_names = img_names[:15]
    
    for img_name in p_img_names:

        img_path = img_folder + img_name
        img = Image.open(img_path).convert('RGB')
        img = img_transform(img).unsqueeze(0).to(device)
        
        seq = sample_sequences(img, model, input_ids, tokenizer)
        encoding = sbert_model.encode([seq], convert_to_tensor=True)
        all_vectors.append(encoding)
        
    all_vectors = torch.cat(all_vectors, dim=0)     # (15, 768)
    
    distances_5.append(get_distances(all_vectors[:5, :]))
    distances_10.append(get_distances(all_vectors[:10, :]))
    distances_15.append(get_distances(all_vectors))
    
    print("\rFinished {}/{} texts".format(i+1, len(t2i)), end='          ')
    
mean5_dist = sum(distances_5) / len(distances_5)
mean10_dist = sum(distances_10) / len(distances_10)
mean15_dist = sum(distances_15) / len(distances_15)

print("RA Text2Image @5: {}".format(mean5_dist))
print("RA Text2Image @10: {}".format(mean10_dist))
print("RA Text2Image @15: {}".format(mean15_dist))    


    