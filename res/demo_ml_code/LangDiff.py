import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Sample Dataset
class TextDataset(Dataset):
    def __init__(self, texts, targets):
        self.texts = texts
        self.targets = targets

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.targets[idx]

# Simple Language Diffusion Model
class LanguageDiffusionModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(LanguageDiffusionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Take the output of the last time step
        return x

# Hyperparameters
vocab_size = 5000  # Example vocabulary size
embed_size = 256   # Embedding size
hidden_size = 512  # Hidden size for LSTM
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# Sample data (replace with your own dataset)
texts = np.random.randint(0, vocab_size, (1000, 10))  # Random integer sequences
targets = np.random.randint(0, vocab_size, (1000,))      # Random targets

# Prepare DataLoader
dataset = TextDataset(texts, targets)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize Model, Loss, and Optimizer
model = LanguageDiffusionModel(vocab_size, embed_size, hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    for i, (text, target) in enumerate(data_loader):
        # Forward pass
        outputs = model(text)
        loss = criterion(outputs, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:  # Print every 10 batches
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(data_loader)}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'language_diffusion_model.pth')
print("Model saved as language_diffusion_model.pth")

# Evaluation (Example)
def evaluate(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():
        for text, target in data_loader:
            outputs = model(text)
            loss = criterion(outputs, target)
            total_loss += loss.item()
    print(f'Average Loss: {total_loss / len(data_loader):.4f}')

# Run evaluation
evaluate(model, data_loader)
