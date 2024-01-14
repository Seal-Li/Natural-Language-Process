import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import utils
import config

class CBOWDataset(Dataset):
    def __init__(self, sentences, word2index, window_size=3):
        self.window_size = window_size

        # 获取训练的数据对
        data = []
        for sentence in sentences:
            indices = [word2index[word] for word in sentence]
            for center in range(len(indices)):
                context = []
                for offset in range(-window_size, window_size + 1):
                    if offset != 0:  # Skip the center word
                        context.append(indices[center + offset] if 0 <= center + offset < len(indices) else word2index['<pad>'])
                data.append((context, indices[center]))
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context_words, center_word = self.data[idx]
        return context_words, center_word


class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, context_words):
        context_embeds = self.embeddings(torch.stack([torch.tensor(word, dtype=torch.long).detach().clone() for word in context_words]))
        context_mean = torch.mean(context_embeds, dim=0, keepdim=True)
        output = self.linear(context_mean)
        return output.squeeze(0)

def train_cbow(model, dataloader, epochs, lr=0.001, log_every=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        hundred_batch_loss = 0.0 # 记录每100个batches的损失
        for i, batch in enumerate(dataloader):
            context_words, center_word = batch

            optimizer.zero_grad()
            output = model(context_words)
            loss = nn.CrossEntropyLoss()(output, center_word)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            total_loss += current_loss
            hundred_batch_loss += current_loss

            # 每100个batches打印一次平均损失
            if (i+1) % log_every == 0:
                avg_hundred_batch_loss = hundred_batch_loss / log_every
                print(f'Average loss for {i+1} batches: {avg_hundred_batch_loss}')
                hundred_batch_loss = 0.0 # 重置hundred_batch_loss

        avg_epoch_loss = total_loss / len(dataloader)
        print(f'Epoch: {epoch}, Loss: {avg_epoch_loss}')


if __name__ == "__main__":
    args = config.argparser()

    sentences_tokens = utils.tokenize_sentences_in_files(args.directory) 
    tokens = [token for sentence in sentences_tokens for token in sentence]
    frequency_dict, total_words = utils.calculate_word_frequencies(tokens)
    encoder_dict = utils.build_encoder_dict(frequency_dict)
    decoder_dict = utils.build_decoder_dict(encoder_dict)

    dataset = CBOWDataset(sentences_tokens, encoder_dict, window_size=args.window_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    model = CBOWModel(total_words, args.embedding_dim).to(device)
    train_cbow(model, dataloader, epochs=args.epochs, lr=args.learning_rate, log_every=args.log_every)

    # 保存模型参数
    torch.save(model.embeddings.state_dict(), 'embeds/cbow/embeddings.pth')
    torch.save(model.linear.state_dict(), 'embeds/cbow/linear.pth')
