import re
import os
import pickle


def save_object(object, name , directory):
    with open(os.path.join(directory, name + '.pkl'), 'wb') as f:
        pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)

def load_object(name , directory):
    with open(os.path.join(directory, name + '.pkl'), 'rb') as f:
        return pickle.load(f)


def isEnglish(sentence):
  sentence = sentence.replace('.', '').replace(' ', '')
  try:
    sentence.encode(encoding='utf-8').decode('ascii')
  except UnicodeDecodeError:
    return False
  else:
    return True


def get_normalized_sentences(text):
  # normalize text and return list of sentences
  ners = re.findall(r'#(\w+)', text)
  sentences = []
  text = text.replace('@', '').replace('#', ' ')
  primary_sentences = text.rsplit('\n')
  for sentence in primary_sentences:
    if isEnglish(sentence): continue
    sentences += [x for x in sentence.rsplit('.') if len(x)>1]
  return sentences, ners


def get_normalized_news(news, ids):
  normalized_news_list = {}
  ners_set = set()
  for id, text in zip(ids, news):
    normalized_sentences, ners = get_normalized_sentences(text)
    normalized_news_list[id] = normalized_sentences
    ners_set.update(ners)
  return normalized_news_list, ners_set