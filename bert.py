import torch
from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F

# from transformers import file_utils
# print(file_utils.default_cache_path)

# sberdevices rubert-large
# tokenizer = BertTokenizer.from_pretrained("ai-forever/ruBert-large")
# model = BertForMaskedLM.from_pretrained("ai-forever/ruBert-large")

# sberdevices rubert-base
tokenizer = BertTokenizer.from_pretrained("ai-forever/ruBert-base")
model = BertForMaskedLM.from_pretrained("ai-forever/ruBert-base")

# deeppavlov
# tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
# model = BertForMaskedLM.from_pretrained("DeepPavlov/rubert-base-cased")

# orig english
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = BertForMaskedLM.from_pretrained("bert-base-uncased")

model.eval()

def predict_masked_sent(text, masked_text_index, top_k=10):
    # Tokenize input
    tokenized_text = tokenizer.tokenize(text)
    print('masking', tokenized_text[masked_text_index])
    tokenized_prompt = ["[CLS]"] + tokenized_text + ["[SEP]"] + tokenized_text[:masked_text_index] + ["[MASK]"] + tokenized_text[masked_text_index+1:] + ["[SEP]"]
    print(tokenized_prompt)
    masked_index = tokenized_prompt.index("[MASK]")
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_prompt)
    tokens_tensor = torch.tensor([indexed_tokens])
    # tokens_tensor = tokens_tensor.to('cuda')    # if you have gpu

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
    top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)

    for i, pred_idx in enumerate(top_k_indices):
        predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        token_weight = top_k_weights[i]
        print("[MASK]: '%s'"%predicted_token, " | weights:", float(token_weight))

# sent = "Настольная [MASK] освещает его мясистое лицо и короткопалые руки." # лампа
sent = "Жанну почитали как мученицу, считали послушной дочерью Римско-католической церкви, символом свободы и независимости."
# sent = "Образ Жанны д’Арк запечатлён в многочисленных произведениях культуры, включая литературу, картины, скульптуры и музыку."

predict_masked_sent(sent, 2)
