import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define the Convolutional Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # (1, 28, 28) -> (16, 14, 14)
            nn.ReLU(True),
            nn.Conv2d(16, 4, 3, stride=2, padding=1),   # (16, 14, 14) -> (4, 7, 7)
            nn.ReLU(True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, 4, stride=2, padding=1),  # (4, 7, 7) -> (16, 14, 14)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),   # (16, 14, 14) -> (1, 28, 28)
            nn.Sigmoid()  # Use sigmoid to get output between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Hyperparameters
batch_size = 64
learning_rate = 1e-3
num_epochs = 10

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
model = ConvAutoencoder().cuda()  # Use .cuda() if using GPU
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data  # Get the image and ignore labels
        img = img.cuda()  # Use .cuda() if using GPU

        # Forward pass
        output = model(img)
        loss = criterion(output, img)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'conv_autoencoder.pth')
print('Model saved.')