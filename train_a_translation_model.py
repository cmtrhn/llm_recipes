import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn.utils.rnn import pad_sequence
import math
from pytorch_lightning.strategies.ddp import DDPStrategy
from datasets import load_dataset
from argparse import ArgumentParser


# Token Embedding
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.scale = math.sqrt(embedding_dim)

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens) * self.scale


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initialize the PositionalEncoding module.

        Args:
            embedding_dim (int): The dimension of the embeddings.
            dropout (float): Dropout probability applied to the positional encoding.
            max_len (int): The maximum sequence length for which positional encodings will be computed.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space
        position = torch.arange(0, max_len).unsqueeze(1)  # Shape: (max_len, 1)
        div_term = torch.exp(- torch.arange(0, embedding_dim, 2) * math.log(10000.0) / embedding_dim)

        pos_encoding = torch.zeros(max_len, embedding_dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        # Register the positional encoding as a buffer so it's not updated during training
        self.register_buffer('pos_encoding', pos_encoding.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for adding positional encoding to the input embeddings.
        Args:
            x (Tensor): Input embeddings of shape (batch_size, seq_len, embedding_dim).
        Returns:
            Tensor: Positional-encoded embeddings of the same shape as input.
        """
        # Add positional encodings to the input tensor
        x = x + self.pos_encoding[:x.size(0), :]
        #x = x + self.pos_encoding[:, :x.size(1), :]
        return self.dropout(x)


# Transformer Model
class Seq2SeqTransformer(LightningModule):
    def __init__(
        self,
        src_vocab_size: int,
        trg_vocab_size: int,
        emb_size: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 1,
        learning_rate: float = 1e-4,
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.save_hyperparameters()

        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.trg_tok_emb = TokenEmbedding(trg_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout)

        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.generator = nn.Linear(emb_size, trg_vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, src, trg, src_mask, trg_mask, src_padding_mask, trg_padding_mask, memory_key_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        trg_emb = self.positional_encoding(self.trg_tok_emb(trg))
        outs = self.transformer(
            src_emb, trg_emb, src_mask, trg_mask, None, src_padding_mask, trg_padding_mask, memory_key_padding_mask
        )
        return self.generator(outs)

    def training_step(self, batch, batch_idx):
        src, trg = batch
        trg_input = trg[:-1, :]
        trg_out = trg[1:, :]

        src_mask, trg_mask, src_padding_mask, trg_padding_mask = self.create_masks(src, trg_input)
        logits = self.forward(src, trg_input, src_mask, trg_mask, src_padding_mask, trg_padding_mask, src_padding_mask)

        loss = self.loss_fn(logits.view(-1, logits.size(-1)), trg_out.view(-1))
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, trg = batch
        trg_input = trg[:-1, :]
        trg_out = trg[1:, :]

        src_mask, trg_mask, src_padding_mask, trg_padding_mask = self.create_masks(src, trg_input)
        logits = self.forward(src, trg_input, src_mask, trg_mask, src_padding_mask, trg_padding_mask, src_padding_mask)

        loss = self.loss_fn(logits.view(-1, logits.size(-1)), trg_out.view(-1))
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def create_masks(self, src, trg):
        src_seq_len = src.size(0)
        trg_seq_len = trg.size(0)

        src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).type(torch.bool)
        trg_mask = self.generate_square_subsequent_mask(trg_seq_len, src.device)

        src_padding_mask = (src == self.hparams.pad_idx).transpose(0, 1)
        trg_padding_mask = (trg == self.hparams.pad_idx).transpose(0, 1)
        return src_mask, trg_mask, src_padding_mask, trg_padding_mask

    @staticmethod
    def generate_square_subsequent_mask(sz, device):
        mask = torch.triu(torch.ones((sz, sz), device=device)) == 1
        mask = mask.transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask


# Dataset Class
class Multi30KDataset:
    def __init__(self):
        self.src_lan = "de"
        self.trg_lan = "en"
        self.tokenizer = {
            self.src_lan: get_tokenizer("spacy", language="de_core_news_sm"),
            self.trg_lan: get_tokenizer("spacy", language="en_core_web_sm"),
        }
        self.special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]
        self.pad_idx = 1
        self.bos_idx = 2
        self.eos_idx = 3

        self.vocab_transform = self.build_vocab()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_vocab(self):
        vocab = {}
        dataset = load_dataset("bentrevett/multi30k", split="train")
        for lang in [self.src_lan, self.trg_lan]:
            vocab[lang] = build_vocab_from_iterator(
                self.yield_tokens(dataset, lang), specials=self.special_symbols, special_first=True
            )
            vocab[lang].set_default_index(0)  # UNK token
        return vocab

    def yield_tokens(self, data_iter, language):
        for data_sample in data_iter:
            yield self.tokenizer[language](data_sample[language])

    def collate_fn(self, batch):
        src_batch, trg_batch = [], []
        src, trg = [pair[self.src_lan] for pair in batch], [pair[self.trg_lan] for pair in batch]
        for src_sample, trg_sample in zip(src, trg):
            src_seq = self.preprocess(src_sample, self.src_lan)
            trg_seq = self.preprocess(trg_sample, self.trg_lan)
            src_batch.append(src_seq)
            trg_batch.append(trg_seq)

        src_batch = pad_sequence(src_batch, padding_value=self.pad_idx)
        trg_batch = pad_sequence(trg_batch, padding_value=self.pad_idx)
        return src_batch, trg_batch

    def preprocess(self, text, lang):
        vocab_token = self.vocab_transform[lang](self.tokenizer[lang](text))
        text_token = torch.cat((torch.tensor([self.bos_idx], dtype=torch.long),
                                torch.tensor(vocab_token, dtype=torch.long),
                                torch.tensor([self.eos_idx], dtype=torch.long)))
        return text_token

    def get_dataloader(self, split="train", batch_size=128):
        dataset = load_dataset("bentrevett/multi30k", split=split)
        return DataLoader(dataset, batch_size=batch_size, collate_fn=self.collate_fn, shuffle=(split == "train"))


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.transformer.encoder(model.positional_encoding(model.src_tok_emb(src)), src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to('cpu')
    for i in range(max_len - 1):
        memory = memory.to('cpu')
        tgt_mask = model.generate_square_subsequent_mask(ys.size(0), 'cpu')
        out = model.transformer.decoder(model.positional_encoding(model.trg_tok_emb(ys)), memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == 3:  # EOS idx
            break
    return ys


def translate(model, dataset, src_sentence):
    model.eval()
    src = dataset.preprocess(src_sentence, 'de').view(-1 ,1)
    num_tokens = len(src)
    src_mask = (torch.zeros(num_tokens, num_tokens))
    trg_tokens = greedy_decode(model, src, src_mask, max_len=num_tokens + 5, start_symbol=2).flatten()
    return " ".join(dataset.vocab_transform['en'].lookup_tokens(list(trg_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")


def infer(ckpt_path):
    dataset = Multi30KDataset()
    model = Seq2SeqTransformer(
        src_vocab_size=len(dataset.vocab_transform["de"]),
        trg_vocab_size=len(dataset.vocab_transform["en"]),
    )
    model.load_state_dict(torch.load(ckpt_path)['state_dict'])
    print(translate(model, dataset, "Eine weiße Katze rennt auf der Straße ."))


# Main Training Loop
if __name__ == "__main__":
    parser = ArgumentParser(description='Language Translation')
    parser.add_argument('--epochs', default=300, type=int, help='number of epochs')
    parser.add_argument('--output', default='./checkpt', type=str, help='folder to save checkpoints')
    parser.add_argument('--checkpoint', default='./checkpt/result.ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--infer', action='store_true', help='infer', default=False, required=False)
    args = parser.parse_args()
    
    dataset = Multi30KDataset()
    train_loader = dataset.get_dataloader(split="train")
    val_loader = dataset.get_dataloader(split="validation")

    model = Seq2SeqTransformer(
        src_vocab_size=len(dataset.vocab_transform["de"]),
        trg_vocab_size=len(dataset.vocab_transform["en"]),
    )
    if not args.infer:
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", filename="tra_mod", save_top_k=1, mode="min", dirpath=args.output)

        trainer = Trainer(
            max_epochs=args.epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            strategy=DDPStrategy(find_unused_parameters=True),
            callbacks=[checkpoint_callback],
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    else:
        infer(args.checkpoint)
