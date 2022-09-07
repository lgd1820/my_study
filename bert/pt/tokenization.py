import os
import re
import sentencepiece as spm

from tqdm import tqdm
from tokenizers import BertWordPieceTokenizer
from konlpy.tag import Mecab
"""
wget https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2
pip install wikiextractor
python -m wikiextractor.WikiExtractor kowiki-latest-pages-articles.xml.bz2
"""

def load_text_data(path):
    doc_pattern = re.compile("^<doc id|</doc>")
    data = []
    for folder in sorted(os.listdir(path)):
        for file_name in sorted(os.listdir(path + f"/{folder}")):
            file_path = path + f"/{folder}/{file_name}"
            with open(file_path, "r") as f:
                while True:
                    line = f.readline()
                    if not line: break
                    if bool(doc_pattern.search(line)): continue
                    line = line.replace("\n", "")
                    if line:
                        data.append(line)
    return data

def save_text_data(data):
    mecab = Mecab()
    with open("kowiki.txt", "w") as f:
        for line in tqdm(data):
            f.write(" ".join(mecab.morphs(line)) + "\n")

def train_setencepiece(corpus, prefix, vocab_size=32000):
    spm.SentencePieceTrainer.Train(
        f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" +  # 7은 특수문자 개수
        " --model_type=unigram" +
        " --max_sentence_length=999999" +  # 문장 최대 길이
        " --pad_id=0 --pad_piece=[PAD]" +  # pad token 및 id 지정
        " --unk_id=1 --unk_piece=[UNK]" +  # unknown token 및 id 지정
        " --bos_id=2 --bos_piece=[BOS]" +  # begin of sequence token 및 id 지정
        " --eos_id=3 --eos_piece=[EOS]" +  # end of sequence token 및 id 지정
        " --user_defined_symbols=[SEP],[CLS],[MASK]" +  # 기타 추가 토큰 SEP: 4, CLS: 5, MASK: 6
        " --input_sentence_size=100000" +  # 말뭉치에서 셈플링해서 학습
        " --shuffle_input_sentence=true")  # 셈플링한 말뭉치 shuffle

def train_bert_tokenizer(corpus, min_frequency, limit_alphabet, vocab_size=32000):
    tokenizer = BertWordPieceTokenizer(strip_accents=False, lowercase=False)
    tokenizer.train(files=[corpus], vocab_size=vocab_size, min_frequency=min_frequency, limit_alphabet=limit_alphabet, show_progress=True)

    tokenizer.save_model("tokenize")

if __name__ == "__main__":
    if not os.path.exists("kowiki.txt"):
        data = load_text_data("text")
        save_text_data(data)
    # train_setencepiece("kowiki.txt", "tokenize/kowiki_32000")
    train_bert_tokenizer("kowiki.txt", 5, 6000)
