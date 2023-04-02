from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

import torch


"""
TODO: PLEASE REMEMBER TO DELETE THIS PACKAGES FROM YOUR MAIN PYTHON ENVIROMENT BEFORE YOU BLOAT IT. 
"""

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')


def tokenize_sentence(sentences):
    tokens = {'input_ids': [], 'attention_mask': []}

    for sentence in sentences:
        # encode each sentence and append to dictionary
        new_tokens = tokenizer.encode_plus(sentence, max_length=128,
                                           truncation=True, padding='max_length',
                                           return_tensors='pt')
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])

    # reformat list of tensors into single tensor
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
    
    return tokens



def text_distance(user_input, text):
    sentences = [text, user_input]

    tokens = tokenize_sentence(sentences)

    outputs = model(**tokens)
    embeddings = outputs.last_hidden_state
    attention_mask = tokens['attention_mask']
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    masked_embeddings = embeddings * mask
    summed = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask
    # encoded_sentence = encoded_sentence.sum(axis=1)

    mean_pooled = mean_pooled.detach().numpy()

    # calculate
    return cosine_similarity(
        [mean_pooled[0]],
        mean_pooled[1:]
    )[0][0]


if __name__ == "__main__":
    inp = "Photosynthesis is the process of feeding plants using the sun to gain energy"
    sol = "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that, through cellular respiration, can later be released to fuel the organism's activities."
    print(text_distance(inp, sol))
