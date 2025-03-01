import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet.requires_grad_(False)  # freeze layers
        
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # remove last layer
        self.proj = nn.Linear(resnet.fc.in_features, n_embd)  

    def forward(self, images):
        features = self.resnet(images).flatten(start_dim=1)  # extract & flatten features
        return self.proj(features)  # project to embedding size

class DecoderRNN(nn.Module):
    def __init__(self, n_embd, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()   
        self.n_embd = n_embd 
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers 

        self.word_embedding = nn.Embedding(vocab_size, n_embd)
        self.lstm = nn.LSTM(n_embd, 
                            hidden_size, 
                            num_layers, 
                            batch_first=True,
                            dropout=0.2)
        
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
    
    def forward(self, features, captions):
        captions = captions[:, :-1] # we dont use last word for prediction
        captions = self.word_embedding(captions) # construct word embeddings (batch_size, seq_length-1, n_embd)
        captions = torch.cat((features.unsqueeze(1), captions), 1) # stick the features as the first input (batch_size, seq_length, n_embd)

        output, _ = self.lstm(captions) # (batch_size, seq_length, hidden_size)
        output = self.fc(output) # (batch_size, seq_length, vocab_size)
        return output[:, 1:, :] # return last word

    def sample(self, features, vocab, max_len=20):
        output = []
        features = features.unsqueeze(0) # unsqueeze the batch dimension

        h = torch.zeros(self.num_layers, 1, self.hidden_size, device=features.device)
        c = torch.zeros(self.num_layers, 1, self.hidden_size, device=features.device)

        for _ in range(max_len):
            x, (h, c) = self.lstm(features, (h, c))
            x = self.fc(x)
            x = x.squeeze(1) # squeeze the sequence dimension
            predict = x.argmax(dim=1)

            if predict.item() == vocab.stoi["<end>"]: # end sampling after end token
                break

            output.append(predict.item())
            features = self.word_embedding(predict.unsqueeze(0))
        
        output = [vocab.itos[i] for i in output] # convert to strings
        return output
