from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
# #Mean Pooling - Take attention mask into account for correct averaging
# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output[0] #First element of model_output contains all token embeddings
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
#
#
# # Sentences we want sentence embeddings for
# sentences = ['质押', '股份股权转让', '投资', '减持', '起诉', '收购', '判决', '签署合同', '担保', '中标']
#
# # Load model from HuggingFace Hub
# tokenizer = AutoTokenizer.from_pretrained('/home/mengfanshen/all-MiniLM-L12-v2/')
# model = AutoModel.from_pretrained('/home/mengfanshen/all-MiniLM-L12-v2/')
#
# # Tokenize sentences
# encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
#
# # Compute token embeddings
# with torch.no_grad():
#     model_output = model(**encoded_input)
#
# # Perform pooling
# sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
#
# # Normalize embeddings
# sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
#
# print("Sentence embeddings:")
# print(sentence_embeddings)
# np.save('schema.npy', sentence_embeddings)

from sentence_transformers import SentenceTransformer

# sentences = ['质押', '股份股权转让', '投资', '减持', '起诉', '收购', '判决', '签署合同', '担保', '中标']
# model = SentenceTransformer('/home/mengfanshen/xiaobu-embedding-v2/')
# embeddings_1 = model.encode(sentences, normalize_embeddings=True)
# print(embeddings_1)
# np.save('schema.npy', embeddings_1)

# sentences = ['质押', '股份股权转让', '投资', '减持', '起诉', '收购', '判决', '签署合同', '担保', '中标']
#
# model = SentenceTransformer('/home/mengfanshen/Dmeta-embedding-zh/')
# embs1 = model.encode(sentences, normalize_embeddings=True)
# print(embs1)
# np.save('schema.npy', embs1)

'''###################################################################'''
import json

from transformers import BertTokenizer, BertModel
import torch

#
# 加载BERT分词器和模型
tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese/')
model = BertModel.from_pretrained('./bert-base-chinese/')

# 要编码的中文句子
sentences = ['质押', '股份股权转让', '投资', '减持', '起诉', '收购', '判决', '签署合同', '担保', '中标']
# sentences = ['实验', '演习', '部署', '支持', '事故', '展览', '冲突', '损伤']
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# 获取模型的输出
with torch.no_grad():
    outputs = model(**inputs)

# 获取句子的编码 (CLS token对应的编码)
sentence_embeddings = outputs.pooler_output

print("Sentence embeddings shape:", sentence_embeddings.shape)  # (batch_size, hidden_size)
np.save('schema.npy', sentence_embeddings)

# 通过sentence-transformer将文本变成tensor
# 生成文本的提示(第一个文本提示和其余文本提示都会用到这个)

tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese/')
model = BertModel.from_pretrained('./bert-base-chinese/')

prompt = {}


def read_text(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            id = json_obj['id']
            sentence = json_obj['prompt']
            inputs = tokenizer(sentence, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**inputs)
            # last_hidden_state = outputs.last_hidden_state
            pooler_output = outputs.pooler_output
            prompt[id] = pooler_output
#
#
# # 要编码的中文句子
read_text('trainAllInformationWithPrompt.json')
read_text('devAllInformationWithPrompt.json')
read_text('testAllInformationWithPrompt.json')
np.save('prompt.npy', prompt)
print('done')

'''###################################################################'''
# # 输出每个句子的编码
# for i, embedding in enumerate(sentence_embeddings):
#     print(f"Sentence {i+1} embedding:", embedding)
