import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from scipy.spatial.distance import cosine


model_name = 'bert-base-multilingual-cased'
emb_model = BertModel.from_pretrained(model_name, output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained(model_name)


def get_subtoken_vectors(sent):
    sent_marked = "[CLS] " + sent + " [SEP]"
    tokenized = tokenizer.tokenize(sent_marked)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized)
    segments_ids = [1] * len(tokenized)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    with torch.no_grad():
        outputs = emb_model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]
    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1,0,2)
    token_vecs_cat = []
    for token in token_embeddings:
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
        token_vecs_cat.append(cat_vec)
    return tokenized, token_vecs_cat


def get_subtokens_to_join(tokenized):
    to_join = []
    join = []
    for i, token in enumerate(tokenized):
        if token.startswith('##'):
            join.append(i)
        else:
            if join:
                join.insert(0, join[0]-1)
                to_join.append(join)
            join = []
    return to_join


def get_full_token(part_for_join, tokenized):
    full_token = ''
    for sub_token in part_for_join:
        full_token += tokenized[sub_token].replace('##', '')
    return full_token


def get_full_token_vectors(sent):
    tokenized_sent, subtoken_vecs = get_subtoken_vectors(sent)
    to_join = get_subtokens_to_join(tokenized_sent)
    merged_join_list = []
    for seq in to_join:
        merged_join_list.extend(seq)

    join_list_ind = 0
    token_vectors = []
    token_words = []
    skip = []
    for i in range(len(subtoken_vecs)):
        if i in skip:
            continue
        if i in merged_join_list:
            join_indices = to_join[join_list_ind]
            token_vec = [np.array(subtoken_vecs[ind]) for ind in join_indices]
            full_token = get_full_token(join_indices, tokenized_sent)
            token_vec_av = np.average(np.array(token_vec), axis=0)
            skip.extend(join_indices)
            join_list_ind += 1
            token_vectors.append(token_vec_av)
            token_words.append(full_token)
        else:
            token_vectors.append(subtoken_vecs[i].numpy())
            token_words.append(tokenized_sent[i])
    return token_words[1:-1], token_vectors[1:-1]


def align_by_cosine_similarity(first_lang_vectors, second_lang_vectors, first_lang_words, second_lang_words):
    aligned = []
    for i, ru_word in enumerate(first_lang_vectors):
        similarities = []
        for word in second_lang_vectors:
            similarity = 1 - cosine(word, ru_word)
            similarities.append(similarity)
        aligned.append([first_lang_words[i], second_lang_words[np.argmax(similarities)], max(similarities)])
    return aligned

