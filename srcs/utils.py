import os
from collections import Counter
import nltk

import config
nltk.download('punkt')


def tokenize_sentences_in_files(directory):
    # 新列表，用于存储所有句子的分词结果
    all_sentences_tokens = []

    # 遍历文件夹下的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            
            # 读取文本文件
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                # 使用nltk对文本进行分句
                sentences = nltk.sent_tokenize(text)
                
                # 对每个句子进行分词
                for sentence in sentences:
                    words = nltk.word_tokenize(sentence)
                    all_sentences_tokens.append(words)
    
    return all_sentences_tokens


def calculate_word_frequencies(tokens):
    frequency_dict = Counter(tokens)
    total_words = sum(frequency_dict.values())
    return frequency_dict, total_words


def build_encoder_dict(frequency_dict, special_chars=['<pad>', '<unk>']):
    # 添加特殊字符
    words = special_chars + list(frequency_dict.keys())
    # 创建编码字典
    encoder_dict = {word: i for i, word in enumerate(words)}
    return encoder_dict


def build_decoder_dict(encoder_dict):
    # 利用编码字典创建解码字典
    decoder_dict = {i: word for word, i in encoder_dict.items()}
    return decoder_dict

if __name__ == "__main__":
    args = config.argparser()

    # 获取所有的句子tokens
    sentences_tokens = tokenize_sentences_in_files(args.directory) 

    # 将所有句子tokens转变为一个大的tokens列表
    tokens = [token for sentence in sentences_tokens for token in sentence]
    # 计算词频和总词数
    frequency_dict, total_words = calculate_word_frequencies(tokens)
    # 创建编码和解码字典
    encoder_dict = build_encoder_dict(frequency_dict)
    decoder_dict = build_decoder_dict(encoder_dict)
    print(encoder_dict)
