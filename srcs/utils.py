import os
import string
import math
from collections import Counter
from nltk.tokenize import word_tokenize

from config import const


def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)

            # 对文档进行分词处理
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read().translate(str.maketrans('', '', string.punctuation)).lower()
                documents.append(word_tokenize(text))
    return documents


def compute_term_frequency(documents):
    term_frequency = Counter()
    for document in documents:
        term_frequency.update(document)
    return term_frequency


def compute_inverse_document_frequency(documents):
    N = len(documents)
    idf = {}
    df = Counter()

    # 计算每一个词的文档频率（DF）
    for document in documents:
        # 对于每个文档，我们只关心一个词是否出现，而不是出现了多少次，所以用set进行去重
        df.update(set(document))
    
    # 计算每一个词的逆文档频率（IDF）
    for term, count in df.items():
        idf[term] = math.log(N / (1 + count))
    
    return idf


def build_encoder_dict(documents):
    # 计算总的词频
    total_term_frequency = compute_term_frequency(documents)

    # 按词频从高到低排序
    sorted_terms = sorted(total_term_frequency.items(), key=lambda item: item[1], reverse=True)

    # 获取所有的单词，并从0开始编码
    encoder_dict = {term: i for i, (term, _) in enumerate(sorted_terms)}

    return encoder_dict


if __name__ == "__main__":
    args = const.argparser()
    documents = load_documents(args.directory)
    term_frequency = compute_term_frequency(documents)
    inverse_document_frequency = compute_inverse_document_frequency(documents)
    encoder_dict = build_encoder_dict(documents)

    print("Term Frequency:", term_frequency)
    print("Inverse Document Frequency:", inverse_document_frequency)
    print("Vocabulary Encoder Dictionary:", encoder_dict)
