import os
from io import open
import torch
import scipy
from scipy import spatial, stats
import torch.nn as nn
import data


path = './data/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt'


device = torch.device("cpu")#torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('./model.pt', 'rb') as f:
    model = torch.load(f).to(device)
#model.eval()

def distance(w1, w2):
    w1_lookup_tensor = torch.tensor([w1], dtype=torch.long, requires_grad=False)
    w2_lookup_tensor = torch.tensor([w2], dtype=torch.long, requires_grad=False)
    #w1_lookup_tensor = w1_lookup_tensor.detach().numpy()
    #w2_lookup_tensor = w2_lookup_tensor.detach().numpy()
    w1_embed = (model.encoder(w1_lookup_tensor).to(device))
    w1_embed = w1_embed.detach().to(device)
    w2_embed = (model.encoder(w2_lookup_tensor).to(device))
    w2_embed = w2_embed.detach().to(device)
    cos = nn.CosineSimilarity()
    pred = cos(w1_embed, w2_embed).to(device)
    #pred = spatial.distance.cosine(w1_embed,w2_embed).to(device)

    return pred



corpus = data.Corpus('./data/wikitext-2')

pred_list = []
actual_sim_list = []
with open(path, 'r', encoding="utf8") as f:
            
    for line in f:
          
        w1, w2, act = line.strip().split('\t')
        if w1 in corpus.dictionary.word2idx and w2 in corpus.dictionary.word2idx:
            w1_idx = corpus.dictionary.word2idx[w1]
            w2_idx = corpus.dictionary.word2idx[w2]
            print(w1)
            print(w2)
            print(act)
            pred = distance(w1_idx, w2_idx)    
            pred_list.append(pred.tolist()[0])
            actual_sim_list.append(float(act))
corr , p_val = stats.spearmanr(pred_list,actual_sim_list)
print(corr)
print('----------')
        

    

