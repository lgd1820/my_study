import torch

from gluonnlp.data import SentencepieceTokenizer
from kobert import get_tokenizer

tok_path = get_tokenizer()
sp = SentencepieceTokenizer(tok_path)

print(sp('한국어 모델을 공유합니다.'))