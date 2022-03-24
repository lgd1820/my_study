import torch
import torch.nn as nn
import pandas as pd
import re
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from module.transformer import Transformer
from torchtext.legacy import data
from soynlp.tokenizer import LTokenizer

tokenizer = LTokenizer()

MAX_LENGTH = 60
D_MODEL = 256
NUM_LAYERS = 7
NUM_HEADS = 8
HIDDEN = 512
BATCH_SIZE = 64
NUM_EPOCH = 100
DROPOUT = 0.3

train_data = pd.read_csv("chatbot.csv")

questions = []
answers = []

for sentence in train_data['Q']:
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    questions.append(sentence)

for sentence in train_data['A']:
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    answers.append(sentence)


Q = data.Field(
    sequential=True,
    use_vocab=True,
    lower=True,
    tokenize=tokenizer,
    batch_first=True,
    init_token="<SOS>",
    eos_token="<EOS>",
    fix_length=MAX_LENGTH
)

A = data.Field(
    sequential=True,
    use_vocab=True,
    lower=True,
    tokenize=tokenizer,
    batch_first=True,
    init_token="<SOS>",
    eos_token="<EOS>",
    fix_length=MAX_LENGTH
)

train_data_set = data.TabularDataset(
    path="chatbot.csv", 
    format="csv", 
    skip_header=False, 
    fields=[("Q",Q), ('A',A)]
)

print('훈련 샘플의 개수 : {}'.format(len(train_data_set)))

Q.build_vocab(train_data_set.Q, train_data_set.A, min_freq = 2)
A.vocab = Q.vocab

VOCAB_SIZE = len(Q.vocab)

PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN = Q.vocab.stoi['<pad>'], Q.vocab.stoi['<SOS>'], Q.vocab.stoi['<EOS>'], Q.vocab.stoi['<unk>']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iter = data.BucketIterator(
    train_data_set, batch_size=BATCH_SIZE,
    shuffle=True, repeat=False, sort=False, device=device
)


print(VOCAB_SIZE)
transformer = Transformer(VOCAB_SIZE, NUM_LAYERS, D_MODEL, NUM_HEADS, HIDDEN, MAX_LENGTH, DROPOUT)

# 네트워크 초기화
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # Liner층의 초기화
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

# 훈련 모드 설정
transformer.train()

# TransformerBlock모듈의 초기화 설정
transformer.apply(weights_init)

print('네트워크 초기화 완료')

# 손실 함수의 정의
criterion = nn.CrossEntropyLoss()

# 최적화 설정
learning_rate = 2e-4
optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate)

best_epoch_loss = float("inf")
epoch_ = []
epoch_train_loss = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 디바이스:", device)
print('-----start-------')
transformer.to(device)

# 네트워크가 어느정도 고정되면 고속화
torch.backends.cudnn.benchmark = True

for epoch in range(NUM_EPOCH):
    epoch_loss = 0.0
    count = 0
    for batch in train_iter:
        questions = batch.Q.to(device)
        answers = batch.A.to(device)
        with torch.set_grad_enabled(True):
            preds = transformer(questions, answers)
            pad = torch.LongTensor(answers.size(0), 1).fill_(PAD_TOKEN).to(device)
            preds_id = torch.transpose(preds,1,2)
            outputs = torch.cat((answers[:, 1:], pad), -1)
            optimizer.zero_grad()
            loss = criterion(preds_id, outputs)  # loss 계산
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 0.5)
            optimizer.step()
            epoch_loss +=loss.item()
            count += 1
    epoch_loss = epoch_loss / count

    if not best_epoch_loss or epoch_loss < best_epoch_loss:
        if not os.path.isdir("snapshot"):
            os.makedirs("snapshot")
        torch.save(transformer.state_dict(), "./snapshot/transformer.pt")
    
    epoch_.append(epoch)
    epoch_train_loss.append(epoch_loss)

    print(f"Epoch {epoch+1}/{NUM_EPOCH} Average Loss: {epoch_loss}")