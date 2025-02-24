import random
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers
transformers.logging.set_verbosity_error()
from transformers import BertTokenizer, BertForPreTraining, AdamW
from datasets import load_dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from argparse import ArgumentParser
# Data Augmentation Utilities
from nltk.corpus import wordnet
import nltk
nltk.download('wordnet')


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

    def forward(self, x):
        """
        Forward pass for adding positional encoding to the input embeddings.
        Args:
            x (Tensor): Input embeddings of shape (batch_size, seq_len, embedding_dim).
        Returns:
            Tensor: Positional-encoded embeddings of the same shape as input.
        """
        x = x + self.pos_encoding[:, :x.size(1), :]
        return self.dropout(x)


# Token Embedding
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.scale = torch.sqrt(torch.tensor(embedding_dim))

    def forward(self, tokens: torch.Tensor):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding.to('device')
        return self.embedding(tokens) * self.scale


class BERTEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim, dropout=0.1):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout)
        self.segment_embedding = nn.Embedding(3, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, bert_inputs, segment_labels=False, train=True):
        my_embeddings = self.token_embedding(bert_inputs.long())
        if train:
            x = self.dropout(
                my_embeddings
                + self.positional_encoding(my_embeddings)
                + self.segment_embedding(segment_labels)
            )
        else:
            x = my_embeddings + self.positional_encoding(my_embeddings)
        return x


def synonym_replacement(sentence, p=0.1):
    """Replace words in a sentence with their synonyms with probability p."""
    words = sentence.split()
    new_words = []
    for word in words:
        if random.random() < p:
            synonyms = wordnet.synsets(word)
            if synonyms:
                synonym = synonyms[0].lemmas()[0].name()
                new_words.append(synonym if synonym != word else word)
            else:
                new_words.append(word)
        else:
            new_words.append(word)
    return " ".join(new_words)


class BERT(torch.nn.Module):

    def __init__(self, vocab_size, d_model=768, n_layers=12, heads=12, dropout=0.1):
        """
        vocab_size: The size of the vocabulary.
        d_model: The size of the embeddings (hidden size).
        n_layers: The number of Transformer layers.
        heads: The number of attention heads in each Transformer layer.
        dropout: The dropout rate applied to embeddings and Transformer layers.
        """
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads
        self.pad_idx = 0

        # Embedding layer that combines token embeddings and segment embeddings
        self.bert_embedding = BERTEmbedding(vocab_size, d_model, dropout)

        # Transformer Encoder layers
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                        nhead=heads,
                                                        dropout=dropout,
                                                        batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)

        # Linear layer for Next Sentence Prediction
        self.next_sentence_head = nn.Linear(d_model, 2)

        # Linear layer for Masked Language Modeling
        self.masked_language_head = nn.Linear(d_model, vocab_size)

    def forward(self, bert_inputs, segment_labels, train=True):
        """
        bert_inputs: Input tokens.
        segment_labels: Segment IDs for distinguishing different segments in the input.
        mask: Attention mask to prevent attention to padding tokens.

        return: Predictions for next sentence task and masked language modeling task.
        """

        padding_mask = (bert_inputs == self.pad_idx).transpose(0, 1)
        # Generate embeddings from input tokens and segment labels
        my_bert_embedding = self.bert_embedding(bert_inputs, segment_labels, train)

        # Pass embeddings through the Transformer encoder
        transformer_encoder_output = self.transformer_encoder(my_bert_embedding, src_key_padding_mask=padding_mask)
        next_sentence_prediction = self.next_sentence_head(transformer_encoder_output[0, :])

        # Masked Language Modeling: Predict all tokens in the sequence
        masked_language_prediction = self.masked_language_head(transformer_encoder_output)

        return next_sentence_prediction, masked_language_prediction


# IMDb Dataset with Data Augmentation
class IMDbBERTDataset(Dataset):
    def __init__(self, split, tokenizer, max_len=128, mask_prob=0.15):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_prob = mask_prob
        imdb_data = load_dataset("imdb", split=split)
        self.texts = imdb_data['text']
        self.sentence_pairs = self._prepare_sentence_pairs()

    def _prepare_sentence_pairs(self):
        sentence_pairs = []
        for text in self.texts:
            # Apply sentence shuffling augmentation
            sentences = text.split('. ')
            random.shuffle(sentences)
            if len(sentences) > 1:
                for i in range(len(sentences) - 1):
                    if random.random() > 0.5:
                        sentence_pairs.append((sentences[i], sentences[i + 1], 1))
                    else:
                        random_text = random.choice(self.texts).split('. ')
                        if random_text:
                            random_sentence = random.choice(random_text)
                            sentence_pairs.append((sentences[i], random_sentence, 0))
        return sentence_pairs

    def _mask_tokens(self, input_ids):
        mlp_labels = input_ids.clone()
        for i in range(len(input_ids)):
            if random.random() < self.mask_prob and input_ids[i] not in \
                    [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id]:
                prob = random.random()
                if prob < 0.8:
                    input_ids[i] = self.tokenizer.mask_token_id
                elif prob < 0.9:
                    input_ids[i] = random.randint(0, self.tokenizer.vocab_size - 1)
        return input_ids, mlp_labels

    def __len__(self):
        return len(self.sentence_pairs)

    def __getitem__(self, idx):
        sent1, sent2, next_sentence_label = self.sentence_pairs[idx]

        # Synonym replacement augmentation
        sent1 = synonym_replacement(sent1)
        sent2 = synonym_replacement(sent2)

        encoded = self.tokenizer(
            sent1,
            sent2,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze(0)
        token_type_ids = encoded['token_type_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        input_ids, mlp_labels = self._mask_tokens(input_ids)

        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'mlp_labels': mlp_labels,
            'next_sentence_label': torch.tensor(next_sentence_label, dtype=torch.long)
        }


# PyTorch Lightning DataModule
class IMDbDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size=16, max_len=128):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.vocab_size = tokenizer.vocab_size
        self.train_dataset, self.val_dataset = None, None

    def setup(self, stage=None):
        self.train_dataset = IMDbBERTDataset(split='train', tokenizer=self.tokenizer, max_len=self.max_len)
        self.val_dataset = IMDbBERTDataset(split='test', tokenizer=self.tokenizer, max_len=self.max_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, drop_last=True)


# PyTorch Lightning Model
class BERTPretrainingModel(pl.LightningModule):
    def __init__(self, vocab_size, lr=5e-5, print_every=10):
        super().__init__()
        self.save_hyperparameters()
        self.the_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert = BERT(vocab_size=vocab_size)
        self.lr = lr
        self.pad_idx = 0
        # The loss function must ignore PAD tokens and only calculates loss for the masked tokens
        self.loss_fn_mlm = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        self.loss_fn_nsp = nn.CrossEntropyLoss()
        self.print_every = print_every

    def forward(self, input_ids, segment_labels, train=True):
        nsp, mlp = self.bert(
            bert_inputs=input_ids,
            segment_labels=segment_labels,
            train=train
        )
        return nsp, mlp

    def training_step(self, batch, batch_idx):
        nsp, mlp = self(
            input_ids=batch['input_ids'],
            segment_labels=batch['token_type_ids'],
        )
        loss = self.get_loss(nsp, mlp, batch['mlp_labels'], batch['next_sentence_label'])
        self.log("train_loss", loss)
        if self.global_step % self.print_every == 0:
            print(f"Step {self.global_step}: Training Loss = {loss.item():.4f}")
        return loss

    def validation_step(self, batch, batch_idx):
        nsp, mlp = self(
            input_ids=batch['input_ids'],
            segment_labels=batch['token_type_ids'],
            train=False
        )
        loss = self.get_loss(nsp, mlp, batch['mlp_labels'], batch['next_sentence_label'])
        self.log("val_loss", loss)
        if batch_idx % self.print_every == 0:
            print(f"Validation Batch {batch_idx}: Loss = {loss.item():.4f}")
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)

    def predict_nsp(self, sentence1, sentence2, tokenizer):
        # Tokenize sentences with special tokens
        tokens = tokenizer.encode_plus(sentence1, sentence2, return_tensors="pt")
        tokens_tensor = tokens["input_ids"].to(self.the_device)
        segment_tensor = tokens["token_type_ids"].to(self.the_device)

        # Predict
        with torch.no_grad():
            nsp_prediction, _ = self.bert(tokens_tensor, segment_tensor)
            # Select the first element (first sequence) of the logits tensor
            first_logits = nsp_prediction[0].unsqueeze(0)  # Adds an extra dimension, making it [1, 2]
            logits = torch.softmax(first_logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()

        # Interpret the prediction
        return "Second sentence follows the first" if prediction == 1 else "Second sentence does not follow the first"

    def predict_mlm(self, sentence, tokenizer):
        inputs = tokenizer(sentence, return_tensors="pt")
        tokens_tensor = inputs.input_ids.to(self.the_device)
        segment_labels = torch.zeros_like(tokens_tensor).to(self.the_device)

        with torch.no_grad():
            output_tuple = self.bert(tokens_tensor, segment_labels)
            predictions = output_tuple[1]  
            mask_token_index = (tokens_tensor == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            predicted_index = torch.argmax(predictions[0, mask_token_index.item(), :], dim=-1)
            predicted_token = tokenizer.convert_ids_to_tokens([predicted_index.item()])[0]
            predicted_sentence = sentence.replace(tokenizer.mask_token, predicted_token, 1)

        return predicted_sentence

    def get_loss(self, nsp, mlp, mlp_labels, nsp_labels):
        next_loss = self.loss_fn_nsp(nsp, nsp_labels)
        mask_loss = self.loss_fn_mlm(mlp.view(-1, mlp.size(-1)), mlp_labels.view(-1))

        loss = next_loss + mask_loss
        return loss


# Main Script
if __name__ == "__main__":
    parser = ArgumentParser(description='BERT Pretraining')
    parser.add_argument('--epochs', default=300, type=int, help='number of epochs')
    parser.add_argument('--output', default='./checkpt', type=str, help='folder to save checkpoints')
    parser.add_argument('--checkpoint', default='./checkpt/result.ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--infer', action='store_true', help='infer', default=False, required=False)
    args = parser.parse_args()
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    data_module = IMDbDataModule(tokenizer=bert_tokenizer, batch_size=128, max_len=128)
    model = BERTPretrainingModel(vocab_size=bert_tokenizer.vocab_size, lr=5e-5)
    if not args.infer:
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", filename="pre_tra", save_top_k=1, mode="min", dirpath=args.output)
        trainer = pl.Trainer(
          max_epochs=args.epochs, 
          accelerator='gpu' if torch.cuda.is_available() else 'cpu',
          strategy=DDPStrategy(find_unused_parameters=True),
          callbacks=[checkpoint_callback],
          )
        trainer.fit(model, data_module)
    else:
        model.load_state_dict(torch.load(args.checkpoint)['state_dict'])
        model.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        # Example usage
        sentence1_nsp = "I went out for shopping to get the missing ingredients."
        sentence2_nsp = "The fish was swimming in the pond."
        
        print(model.predict_nsp(sentence1_nsp, sentence2_nsp, bert_tokenizer))
        
        # Example usage
        sentence_mask = "I went out for [MASK] to get the missing ingredients."
        print(model.predict_mlm(sentence_mask, bert_tokenizer))
