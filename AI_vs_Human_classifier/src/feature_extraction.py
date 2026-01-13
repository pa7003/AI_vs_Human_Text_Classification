from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

def extract_features(texts, model_name='bert-base-uncased', max_length=128, device='cpu'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    features = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        features.append(cls_embedding.squeeze())
    return np.array(features)
