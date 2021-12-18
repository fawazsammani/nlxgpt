import json
import torch
from bert_score import score
from cococaption.pycocotools.coco import COCO
from cococaption.pycocoevalcap.eval import COCOEvalCap

annotations_path = 'nle_data/VCR/vcr_test.json'
pred_unf_full_path = 'cococaption/results/unfiltered_captions_full_8.json'   # explanations + answers
pred_unf_exp_path = 'cococaption/results/unfiltered_captions_full_8.json'    # explanations
save_filtered_caps = 'cococaption/results/vcr_filtered_results.json'
save_filtered_scores = 'cococaption/results/vcr_filtered_scores.json'
annTest = 'cococaption/annotations/vcr_test_annot_exp.json'
keep_keys_path = 'cococaption/results/correct_keys.json'
threshold = 0.92

gt = json.load(open(annotations_path, 'r'))
prd = json.load(open(pred_unf_full_path, 'r'))

predictions = {}
for item in prd:
    predictions[item['image_id']] = item['caption'].split("because")[0].strip()
    
ground_truths = {}
for key,value in gt.items():
    ground_truths[int(key)] = value['answers']
    
refs = []
cands = []
all_keys = []

for key,value in predictions.items():
    all_keys.append(key)
    refs.append(ground_truths[key].lower())
    cands.append(value.lower())


out, hash = score(cands, refs, model_type='distilbert-base-uncased', verbose=True, idf=False, lang="en", return_hash=True)
P, R, F1 = out

print("Accuracy: ", F1.mean())

all_keys = torch.LongTensor(all_keys)

with open(keep_keys_path, 'w') as w:
    json.dump(all_keys[F1 >= threshold].tolist(), w)
    

correct_keys = json.load(open(keep_keys_path, 'r'))
exp_predictions = json.load(open(pred_unf_exp_path, 'r'))
exp_preds = [item for item in exp_predictions if item['image_id'] in correct_keys]

with open(save_filtered_caps, 'w') as w:
    json.dump(exp_preds, w)

coco = COCO(annTest)
cocoRes = coco.loadRes(save_filtered_caps)
cocoEval = COCOEvalCap(coco, cocoRes)
cocoEval.params['image_id'] = cocoRes.getImgIds()
cocoEval.evaluate()

with open(save_filtered_scores, 'w') as w:
    json.dump(cocoEval.eval, w)

