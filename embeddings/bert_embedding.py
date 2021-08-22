import os
import argparse
import logging
import tqdm
import pandas as pd
import numpy as np

from transformers import BertTokenizer, TFBertModel, pipeline

from ..utils.utils import get_normalized_news, load_object, save_object


def load_pretrained_model(address):
    tokenizer = BertTokenizer.from_pretrained(address)
    model = TFBertModel.from_pretrained(address, output_hidden_states = True)
    return model, tokenizer

def update_tokenizer_and_model(tokenizer, model, ners_set):
    tokenizer.add_tokens(ners_set)
    model.resize_token_embeddings(len(tokenizer)) 
    return tokenizer, model

def embedding(sentence, model, tokenizer):
    pipe = pipeline('feature-extraction', model=model, tokenizer=tokenizer)
    features = pipe(sentence)
    features = np.squeeze(features)
    
    return features

def embedding(sentence, model, tokenizer):
    pipe = pipeline('feature-extraction', model=model, tokenizer=tokenizer)
    features = pipe(sentence)
    features = np.squeeze(features)
    return features


def update_embeddings(normalized_news, ids, model, tokenizer, output_path, batch_size=1000):
    out_dir = os.path.join(output_path, 'embedding')

    try: embedding_news = load_object(name='embedding_news', directory=out_dir)
    except: embedding_news = {}
    try: key_list = load_object(name='key_list', directory=out_dir)
    except: key_list = []

    start = len(key_list)
    print('key_list size {}'.format(start))
    new_ids = list(set(ids) - set(key_list))

    for i, id in tqdm(enumerate(new_ids, start=start), total=len(new_ids)):
        if id in key_list: continue
        embedded_news_all_sentences = []
        for sentence in normalized_news[id]:
            sentence += ' .'
            embedded_sentence = embedding(sentence, model, tokenizer)
            embedded_news_all_sentences.extend(embedded_sentence)
        embedding_news[id] = embedded_news_all_sentences
        key_list.append(id)

        if i % batch_size == 0:
            print('save step {}'.format(i))
            save_object(embedding_news, name='embedding_news_{}'.format(i), directory=out_dir)
            save_object(key_list, name='key_list', directory=out_dir)
            embedding_news = {}
    save_object(embedding_news, name='embedding_news', directory=out_dir)
    save_object(key_list, name='key_list', directory=out_dir)


def load_aggregate_embeddings(name, length, output_path, steps=1000, offset=500):
    out_dir = os.path.join(output_path, 'embedding')
    embedding_news = load_object(name='{}'.format(name), directory=out_dir)
    for i in tqdm(range(0, length + offset, steps)):
        step_emb = load_object(name='{}_{}'.format(name, i), directory=out_dir)
        embedding_news.update(step_emb)
    return embedding_news


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text Embedding Using Bert')
    parser.add_argument('--pre-train-address', 
                        required=False, 
                        default='HooshvareLab/bert-fa-zwnj-base', 
                        type=str, 
                        help='pretrained hugging face models address')
    parser.add_argument('--output', 
                        required=False, 
                        default='.', 
                        type=str, 
                        help='path to output folder')
    parser.add_argument('--input', 
                        required=False, 
                        default='.', 
                        type=str, 
                        help='path to news csv')
    parser.add_argument('--batch-size',
                        required=False, 
                        default=1000, 
                        type=int, 
                        help='save batch size - for big datasets')
    
    args = parser.parse_args()

    address = args.pre_train_address
    output_path = os.path.abspath(args.output)
    input_path = os.path.abspath(args.input)
    batch_size = args.batch_size

    news_df = pd.read_csv(input_path, index_col=None)
    news = list(news_df['news'])
    ids = list (news_df['id'])

    normalized_news, ners_set = get_normalized_news(news)

    model, tokenizer = load_pretrained_model(address)
    tokenizer, model = update_tokenizer_and_model(tokenizer, model, ners_set)

    update_embeddings(
        normalized_news=normalized_news, 
        ids=ids, 
        model=model, 
        tokenizer=tokenizer,
        output_path=output_path,
        batch_size=batch_size)

