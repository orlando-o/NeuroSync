import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data import Field, BucketIterator
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Define fields
TEXT = Field(tokenize=tokenizer.encode, init_token='<s>', eos_token='</s>', lower=True)
LABEL = Field(sequential=False)

# Load the IMDB dataset
train_data, test_data = IMDB.splits(TEXT, LABEL)

# Build vocabulary
TEXT.build_vocab(train_data, max_size=10000, min_freq=10)
LABEL.build_vocab(train_data)

# Create iterators
train_iterator, test_iterator = BucketIterator.splits((train_data, test_data), batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu')

# Define the transformer model
class LanguageTransformer(nn.Module):
    def __init__(self):
        super(LanguageTransformer, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.logits

# Initialize model, optimizer, and loss function
model = LanguageTransformer().to('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

# Training loop
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    
    for batch in tqdm(iterator):
        optimizer.zero_grad()
        input_ids = batch.text.T  # Transpose to shape (batch_size, seq_length)
        labels = input_ids.clone()  # Labels are the same as input for language modeling

        # Move data to the appropriate device
        input_ids = input_ids.to('cuda' if torch.cuda.is_available() else 'cpu')
        labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')

        # Forward pass
        outputs = model(input_ids)
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        epoch_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    
    return epoch_loss / len(iterator)

# Evaluation loop
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(iterator):
            input_ids = batch.text.T
            labels = input_ids.clone()
            input_ids = input_ids.to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')

            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

# Training the model
for epoch in range(5):
    train_loss = train(model, train_iterator, optimizer, criterion)
    test_loss = evaluate(model, test_iterator, criterion)
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f}')
