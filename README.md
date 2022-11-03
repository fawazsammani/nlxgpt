# [NLX-GPT](https://arxiv.org/abs/2203.05081)
Official Code for **NLX-GPT: A Model for Natural Language Explanations in Vision and Vision-Language Tasks** <br>
[arXiv](https://arxiv.org/abs/2203.05081) | [video](https://www.youtube.com/watch?v=GwgFYUZKvRk&ab_channel=ArtificialIntelligence)
<br>
<br>
**[NEW]** Gradio web-demo for VQA-X [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Fawaz/nlx-gpt)
<br>
**[NEW]** Gradio web-demo for ACT-X [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Fawaz/nlx_gpt_action)

<br>
<br>
<p align="center">
<img src="utils/demo.png" width="512"/>
  </p>

### Requirements
- [PyTorch](https://pytorch.org/) 1.8 or higher
- [CLIP](https://github.com/openai/CLIP) (install with `pip install git+https://github.com/openai/CLIP.git`)
- [transformers](https://huggingface.co/docs/transformers/index) (install with `pip install transformers`)
- [accelerate](https://huggingface.co/docs/accelerate/index.html) for distributed training (install with `pip install git+https://github.com/huggingface/accelerate`)

### Images Download
We conduct experiments on 4 different V/VL NLE Datasets: **VQA-X, ACT-X, e-SNLI-VE** and **VCR**. Please download the images into a folder in your directory named `images` using the following links (our code *does not* use pre-cached visual features. Instead, the features are extracted directly during code execution):
<br>
- **VQA-X**: [COCO](https://cocodataset.org/#download) `train2014` and `val2014` images<br>
- **ACT-X**: [MPI](http://human-pose.mpi-inf.mpg.de/#download) images. Rename to `mpi` <br>
- **e-SNLI-VE**: [Flickr30K](http://shannon.cs.illinois.edu/DenotationGraph/) images. Rename to `flickr30k` <br>
- **VCR**: [VCR](https://visualcommonsense.com/download/) images. Rename to `vcr` <br>

### Annotations Download
We structure the annotations for the NLE datasets. You can dowloaded the structured annotations from here: [VQA-X](https://drive.google.com/drive/folders/16sJjeEQE2o23G-GGUi870ubXzJjdRDua?usp=sharing), [ACT-X](https://drive.google.com/drive/folders/1FffVDEgHmqnWiqD5-B5700gqErQ-3U1M?usp=sharing), [e-SNLI-VE](https://drive.google.com/drive/folders/16YyIbjOr0XAD-34sUFsmrsXxbD5aKTVf?usp=sharing), [VCR](https://drive.google.com/drive/folders/1Cpk0wngnnlW0zr_dfHvdR15Lec56HSZm?usp=sharing). Place them in `nle_data/dataset_name/` directory. `dataset_name` can be `{VQA-X, ACT-X, eSNLI-VE, VCR}`. The pretraining annotations are [here](https://drive.google.com/drive/folders/130kPrux7vmv8jxC39m2ceeH2b6lPm_xU?usp=sharing). Please see [this](https://github.com/fawazsammani/nlxgpt/issues/6) issue also for clarification on which pretrain annotations to use. The code to preprocess the annotations can be found in `utils/nle_preprocess.ipynb`. 
<br>
<br>
You also need [cococaption](https://github.com/tylin/coco-caption) and the annotations in the correct format in order to perform evaluation on NLG metrics. 
We use the cococaption python3 toolkit [here](https://github.com/ruotianluo/coco-caption/tree/ea20010419a955fed9882f9dcc53f2dc1ac65092). Please download it and place the `cococaption` folder in your directory. The annotations in the correct format can be downloaded [here](https://drive.google.com/drive/folders/1b8kUPbgtEduiz8A_VbUg0W_vca7PyXsZ?usp=sharing). Please place them in the `annotations` folder. The code to convert the natural language explanations data from the source to the format that cococaption expects for evaluation can be found in `utils/preprocess_for_cococaption_eval.ipynb`.

You will also need [BertScore](https://github.com/Tiiiger/bert_score) if you evaluate using it. You may install with `pip install bert_score==0.3.7` <br>

### Code
1 GPU is enough for finetuning on NLE. However if you wish to do distributed training, please setup first using `accelerate`. Note that you can still use `accelerate` even if you have 1 GPU. In your environment command line, type: <br>
```bash
accelerate config
```
and answer the questions. <br>
##### VQA-X 
Please run from the command line with: <br>
```bash
accelerate launch vqaX.py
```
Note: To finetune from the pretrained captioning model, please set the `finetune_pretrained` flag to `True`. 
##### ACT-X 
Please run from the command line with: <br>
```bash
accelerate launch actX.py
```
Note: To finetune from the pretrained captioning model, please set the `finetune_pretrained` flag to `True`. 

##### e-SNLI-VE
Please run from the command line with: <br>
```bash
accelerate launch esnlive.py
```

##### e-SNLI-VE (+ Concepts)
Please run from the command line with: <br>
```bash
accelerate launch esnlive_concepts.py
```

##### VCR
Please run from the command line with: <br>
```bash
accelerate launch vcr.py
```
This will give you the unfiltered scores. After that, we use BERTScore to filter the incorrect answers and get the filtered scores (see paper Appendix for more details). Since BERTScore takes time to calculate, it is not ideal to run it and filter scores after every epoch. Therefore, we perform this operation once on the epoch with the best unfiltered scores. Please run:
```bash
python vcr_filter.py
```

### Models
All models can be downloaded from the links below:
- Pretrained Model on Image Captioning: [link](https://drive.google.com/drive/folders/1_3xVoMJwV98j7vUxEEJsQ0LGqFGtfkca?usp=sharing)
- VQA-X (w/o pretraining): [link](https://drive.google.com/drive/folders/187_WSQUSHNf1Ga9qrynbUR98jMlwl3NF?usp=sharing)
- VQA-X (w/ pretraining): [link](https://drive.google.com/drive/folders/1Bfc__0HRzYPyvRe0Ur_oSbhO8dSavT4e?usp=sharing)
- ACT-X (w/o pretraining): [link](https://drive.google.com/drive/folders/1b9fG54lm-PnXrPvYhnFe4T78gHrU93IS?usp=sharing)
- ACT-X (w/ pretraining): [link](https://drive.google.com/drive/folders/1oiPm9f5I7ZmvMVxkq9crCSH02qxizZ7_?usp=sharing)
- Concept Head + Wordmap (used in e-SNLI-VE w/ concepts): [link](https://drive.google.com/drive/folders/1Hnk5NVvP5SqC-DeJT-znqwGzpU796QQl?usp=sharing)
- e-SNLI-VE (w/o concepts): [link](https://drive.google.com/drive/folders/1A4NlhIWy5byrqEfbIeh7Mgdxh1WGOD2x?usp=sharing)
- e-SNLI-VE (w/ concepts): [link](https://drive.google.com/drive/folders/1q4C9jujdHgXkc5IEsBD1HxAiVm4f8zp1?usp=sharing)
- VCR: [link](https://drive.google.com/drive/folders/1ApplfjJjQ-eLz8zjcf4iT1OSs0mY1dmk?usp=sharing)<br>

Note: Place the concept model and its wordmap in a folder: `pretrained_model/`

### Results 
The output results (generated text) on the test dataset can be downloaded from the links below. `_filtered` means that the file contains only the explanations for which the predicted answer is correct. 
`_unfiltered` means that all the explanations are included, regardless of whether the predicted answer is correct or not. 
`_full` means the full output prediction (inclusing the answer + explanation). `_exp` means the explanation part only. All evaluation is performed on `_exp`. 
See section 4 of the paper for more details. 
- VQA-X (w/o pretraining): [link](https://drive.google.com/drive/folders/10TR-cWJCGauU9i7FOAQWp2N3XTNWM_V6?usp=sharing)
- VQA-X (w/ pretraining): [link](https://drive.google.com/drive/folders/1nipKCftK2uSfBarrIrQYCnylpje8G9W_?usp=sharing)
- ACT-X (w/o pretraining): [link](https://drive.google.com/drive/folders/1vQN6rAzHGU12ikxKe7e4dGzm1okpZG9a?usp=sharing)
- ACT-X (w/ pretraining): [link](https://drive.google.com/drive/folders/1c_mlTc9HH_P0qMcu-mnQXAglbDNP2mNw?usp=sharing)
- e-SNLI-VE (w/o concepts): [link](https://drive.google.com/drive/folders/1rfgYyf9-8N2d3Jk-H6jWLtV0ii3tLqEF?usp=sharing)
- e-SNLI-VE (w/ concepts): [link](https://drive.google.com/drive/folders/1ex8JXxFF9D02WlI6qkCmryxmQXAlNKX_?usp=sharing)
- VCR: [link](https://drive.google.com/drive/folders/1Fp1xHux3GD8qdg7a2FQBHDdnPbw6MQL6?usp=sharing)

Please note that in case of VCR, the results shown in Page 4 of the appendix may not identically correspond to the results and pretrained model in the links above. We have trained several models and randomly picked one for presenting the qualitative results. 

### Proposed Evaluation Metrics
Please see `explain_predict` and `retrieval_attack` folders.
