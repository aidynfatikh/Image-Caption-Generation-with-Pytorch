import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import pickle

import model as models
import get_train_data

import matplotlib.pyplot as plt

# MAKE SURE FOLDERS EXIST
os.makedirs("dataset", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# HYPERPARAMETERS
batch_size = 32
hidden_size = 256
num_layers = 2
n_embd = 196
lr = 1e-3
num_epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# LOADING THE DATASET
image_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
captions_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"

images_zip_path = "dataset/Flickr8k_Dataset.zip"
captions_zip_path = "dataset/Flickr8k_text.zip"

transform = transforms.Compose([
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

get_train_data.load_data(image_url, captions_url, images_zip_path, captions_zip_path)

vocab = get_train_data.Vocabulary(min_freq=1)
caption_dict, max_caption_length, all_captions = get_train_data.get_caption_dict(get_train_data.read_captions(captions_zip_path))
vocab.build_vocabulary(all_captions)
vocab_size = len(vocab.stoi)

dataset = get_train_data.Flickr8k(images_zip_path, captions_zip_path, transform, vocab)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

# SAVE THE VOCABULARY
with open("dataset/vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

# INITIALIZE THE MODEL
encoder = models.EncoderCNN(n_embd).to(device)
decoder = models.DecoderRNN(n_embd, hidden_size, vocab_size, num_layers).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=vocab.stoi["<pad>"]) # ignore <pad> token to remove its contribution to the loss
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr) 
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

if False:  # TWEAK THIS IF YOU HAVE CHECKPOINTS
    epoch = 1 # checkpoint you want to load
    encoder.load_state_dict(torch.load(f"checkpoints/encoder_{epoch}.pth", weights_only=False))
    decoder.load_state_dict(torch.load(f"checkpoints/decoder_{epoch}.pth", weights_only=False))

# TRAINING LOOP
for epoch in range(num_epochs):
    encoder.train()
    decoder.train()

    total_loss = 0
    batch_iter = 1

    for image, caption in dataloader:
        # move data to device
        image, caption = image.to(device), caption.to(device)

        # zero the gradients
        decoder.zero_grad()
        encoder.zero_grad()
        
        # get generated captions
        features = encoder(image)
        features = decoder(features, caption)
        
        # compare generated with actual captions
        loss = criterion(features.reshape(-1, vocab_size), caption[:, 1:].reshape(-1, ))

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # monitor loss every while
        if (batch_iter%100 == 0 or batch_iter == len(dataloader) or batch_iter == 1):
            print(f"Epoch: {epoch+1}/{num_epochs}\tStep: {batch_iter}/{len(dataloader)}\tLoss: {(total_loss / batch_iter):.4f}")

        batch_iter += 1

    # save model for current epoch
    torch.save(encoder.state_dict(), f"checkpoints/encoder_{epoch+1}.pth")
    torch.save(decoder.state_dict(), f"checkpoints/decoder_{epoch+1}.pth")
    print(f"Epoch: {epoch+1}/{num_epochs}\tAvgLoss: {(total_loss / batch_iter):.4f}")

    scheduler.step()