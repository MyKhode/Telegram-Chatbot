<div align="center">

# 🏅Telegram Khmer Chatbot + Open Source Code🏅

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
<img alt="GitHub forks" src="https://img.shields.io/github/forks/SOYTET/Telegram-Chatbot">
<img alt="GitHub License" src="https://img.shields.io/github/license/SOYTET/Telegram-Chatbot">
<img alt="GitHub Release" src="https://img.shields.io/github/v/release/SOYTET/Telegram-Chatbot">
<img alt="GitHub code size in bytes" src="https://img.shields.io/github/languages/code-size/SOYTET/Telegram-Chatbot">


intents.json
   ↓
Read patterns & tags
   ↓
Khmer Tokenization (khmernltk.word_tokenize)
   ↓
Build Vocabulary (all_words)
   ↓
Bag of Words Encoding
   ↓
X_train (features) + y_train (labels)
   ↓
PyTorch Dataset + DataLoader
   ↓
Neural Network (SimpleNet)
   ↓
Training Loop (Forward → Loss → Backprop → Update)
   ↓
Save model weights + metadata (data.pth)


</div>

### This will never happen if Khmer NLTK Module doesn't exist, thanks 🙏
## 🎯TODO

- [X] Tokenization and Bag of Words Creation
- [X] Data Preprocessing
- [X] Neural Network Model
- [X] Training the Model
- [X] Model Evaluation
- [X] Telegram Bot Integration
- [ ] Make it more inteligent and flexible

## 💪Installation

```bash
pip install numpy
```
```bash
pip install khmer-nltk
```
```bash
pip install python-telegram-bot
```
```bash
pip install torch
```

## 🏹 Quick tour

[[Official Blog Post Website Url]](https://www.ikhode.site/posts/khmer-telegram-chatbot/)

To get the quick way of using this code package, please install requirement.txt  and read the comment in code header before run that code script. 
#### Some Mention
- Check folder directory 
- create virtual environment with "myEnv" folder
- install with requrements version module
- Check Bot Api 
- Check Bot Username
- Make sure u aready istall library I mentioned.
(if u have any issue, U can contact me by Telegram Directly @SOYTET)

## 🔯 Demo

#### Run the ChatApp Scripts to Excute Telegram integration and Bot response after training finish
```bash
Python ./App/ChatApp.py
```
![Screenshot 2024-05-28 180135](https://github.com/SOYTET/Telegram-Chatbot/assets/132768132/422ad075-38b3-45aa-b39d-9544491f9bea)


#### While Running ChatApp.py Script, so u can make some chat with Telegram Bot which u integrate with your own
```bash
សួស្ដីបង
```
![Screenshot 2024-05-28 180224 (1)](https://github.com/SOYTET/Telegram-Chatbot/assets/132768132/d7462838-f301-4723-b52a-aa4efe63e204)


### ✍️ Citation

```bibtex
@misc{Chatbot,
  author = {SOY TET},
  title = {Khmer Telegram Traditional Chatbot},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository}
}
```
#### Used in:
- [stopes: A library for preparing data for machine translation research](https://github.com/facebookresearch/stopes)
- [LASER Language-Agnostic SEntence Representations](https://github.com/facebookresearch/LASER)
- [Pretrained Models and Evaluation Data for the Khmer Language](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9645441)
- [Multilingual Open Text 1.0: Public Domain News in 44 Languages](https://arxiv.org/pdf/2201.05609.pdf)
- [ZusammenQA: Data Augmentation with Specialized Models for Cross-lingual Open-retrieval Question Answering System](https://arxiv.org/pdf/2205.14981.pdf)
- [Shared Task on Cross-lingual Open-Retrieval QA](https://www.aclweb.org/portal/content/shared-task-cross-lingual-open-retrieval-qa)
- [No Language Left Behind: Scaling Human-Centered Machine Translation](https://research.facebook.com/publications/no-language-left-behind/)
- [Wordless](https://github.com/BLKSerene/Wordless)
- [A Simple and Fast Strategy for Handling Rare Words in Neural Machine Translation](https://aclanthology.org/2022.aacl-srw.6/)

### 👨‍🎓 References

- [NLP: Text Segmentation Using Conditional Random Fields](https://medium.com/@phylypo/nlp-text-segmentation-using-conditional-random-fields-e8ff1d2b6060)
- [Khmer Word Segmentation Using Conditional Random Fields](https://www2.nict.go.jp/astrec-att/member/ding/KhNLP2015-SEG.pdf)
- [Word Segmentation of Khmer Text Using Conditional Random Fields](https://medium.com/@phylypo/segmentation-of-khmer-text-using-conditional-random-fields-3a2d4d73956a)
