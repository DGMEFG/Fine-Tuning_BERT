import torch
from tqdm.auto import tqdm
from transformers import get_scheduler
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=5).to(device)
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased')


class Data(Dataset):
    def __init__(self, path, train=True):
        self.sentences = []
        self.labels = []
        i = 0
        with open (path, 'r') as f:
            for line in f:
                if i:
                    _ = list(line.split('\t'))
                    self.sentences.append(_[2])
                    if train:
                        self.labels.append(int(_[3]))
                    else:
                        self.labels.append(int(_[0]))
                i += 1
                if i > 10:
                  break
        self.data = tokenizer(self.sentences, padding=True, return_tensors='pt')['input_ids']
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx] if len(self.labels) else None

data_iter = DataLoader(Data("./train.tsv"), batch_size=256, shuffle=True)
optimizer = AdamW(model.parameters(), lr=1e-5)


num_epochs = 100
num_training_steps = num_epochs * len(data_iter)
lr_sheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

model.train()
progress_bar = tqdm(range(num_epochs), desc="Epoch")
for epoch in range(num_epochs):
    sum = 0
    for xi, yi in data_iter:
        optimizer.zero_grad()
        xi, yi = xi.to(device), yi.to(device)
        loss = model(xi, labels=yi).loss
        loss.backward()
        sum += loss.item()
        optimizer.step()
        lr_sheduler.step()
    progress_bar.update(1)
    print(f"loss: {sum/len(data_iter)}")

test = Data("./test.tsv", train=False)

model.eval()
with open('ans.txt', 'w') as f:
  for i, phraseid in enumerate(test.labels):
      f.write(f"{phraseid}\t{model(test.data[i].unsqueeze(0).to(device)).logits.argmax().item()}\n")
    