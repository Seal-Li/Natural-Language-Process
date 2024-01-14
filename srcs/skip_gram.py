import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import utils
import config

class SkipGramDataset(Dataset):
    def __init__(self, sentences, word2index, window_size=3, num_neg_samples=10):
        self.window_size = window_size
        self.num_neg_samples = num_neg_samples
        
        self.word_counts = torch.ones(len(word2index), dtype=torch.float32)  # 先假设所有单词次数均为1
        
        # 获取训练的数据对
        data = []
        for sentence in sentences:
            indices = [word2index[word] for word in sentence]
            for center in range(len(indices)):
                for offset in range(-window_size, window_size + 1):
                    context = center + offset
                    if context < 0 or context >= len(indices) or context == center:
                        continue
                    data.append((indices[center], indices[context]))
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        center_word, context_word = self.data[idx]
        neg_samples = torch.multinomial(self.word_counts, self.num_neg_samples, replacement=True)
        return center_word, context_word, neg_samples


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(SkipGramModel, self).__init__()
        self.center_embeds = nn.Embedding(vocab_size, embed_dim)
        self.context_embeds = nn.Embedding(vocab_size, embed_dim)

    def forward(self, center_word, context_word, neg_samples):
        center_embed = self.center_embeds(center_word)
        context_embed = self.context_embeds(context_word)

        # 正例的损失
        correct_log_bdot = torch.bmm(
            center_embed.view(center_word.shape[0], 1, -1), 
            context_embed.view(center_word.shape[0], -1, 1)).sigmoid().log()

        # 负例的损失
        neg_embed = self.context_embeds(neg_samples)
        incorrect_bdot = torch.bmm(
            neg_embed, 
            center_embed.unsqueeze(2)).sigmoid().log()

        return -(correct_log_bdot + incorrect_bdot.sum(1))


def train(model, dataloader, epochs, lr=0.001, log_every=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        hundred_batch_loss = 0.0 # 记录每100个batches的损失
        for i, batch in enumerate(dataloader):
            center_word, context_word, neg_samples = batch

            optimizer.zero_grad()
            loss = model(center_word, context_word, neg_samples).mean()
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            total_loss += current_loss
            hundred_batch_loss += current_loss

            if (i+1) % log_every == 0:
                avg_hundred_batch_loss = hundred_batch_loss / log_every
                print(f'Average loss for {i+1} 100 batches: {avg_hundred_batch_loss}')
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

    vocab_size = total_words

    dataset = SkipGramDataset(sentences_tokens, encoder_dict, window_size=args.window_size, num_neg_samples=args.num_neg_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    model = SkipGramModel(vocab_size, args.embedding_dim).to(device)
    train(model, dataloader, epochs=args.epochs, lr=args.learning_rate, log_every=args.log_every)

    # 在训练结束后保存模型参数
    torch.save(model.center_embeds.state_dict(), 'embeds/skip_gram/center_embeds.pth')
    torch.save(model.context_embeds.state_dict(), 'embeds/skip_gram/context_embeds.pth')
