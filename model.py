import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, n_embd):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights="ResNet50_Weights.DEFAULT")
        for param in resnet.parameters(): # we dont want to train the feature extractor
            param.requires_grad_(False)
        
        # extract the components of ResNet excluding the last layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules) 
        self.proj = nn.Linear(resnet.fc.in_features, n_embd)

    def forward(self, images):
        features = self.resnet(images) 
        features = features.view(features.size(0), -1)
        features = self.proj(features) # (batch_size, resnet.fc.in_features) -> (batch_size, n_embd)
        return features
    

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
        caption_embed = self.word_embedding(captions[:, :-1]) # construct word embeddings (batch_size, seq_length-1, n_embd)
        caption_embed = torch.cat((features.unsqueeze(dim=1), caption_embed), 1) # stick the features as the first input (batch_size, seq_length, n_embd)

        output, _ = self.lstm(caption_embed) # (batch_size, seq_length, hidden_size)
        output = self.fc(output) # (batch_size, seq_length, vocab_size)
        return output[:, 1:, :] # return last word

    def sample(self, inputs, vocab, states=None, max_len=20):
        output = []
        inputs = inputs.unsqueeze(0) # unsqueeze the batch dimension
        (h, c) = (torch.zeros(self.num_layers, 1, self.hidden_size).to(inputs.device), torch.zeros(self.num_layers, 1, self.hidden_size).to(inputs.device))
        for _ in range(max_len):
            x, (h, c) = self.lstm(inputs, (h, c))
            x = self.fc(x)
            x = x.squeeze(1) # squeeze the sequence dimension
            predict = x.argmax(dim=1)
            if predict.item() == vocab.stoi["<end>"]: # end sampling after end token
                break

            output.append(predict.item())
            inputs = self.word_embedding(predict.unsqueeze(0))
        
        output = [vocab.itos[i] for i in output] # convert to strings
        return output