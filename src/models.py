from torch import nn
from collections import OrderedDict

class LinearBlock(nn.Module):
    
    def __init__(self, layer_sequence, add_relu=False):
        super(LinearBlock, self).__init__()
        num_layers = len(layer_sequence) - 1
        layers = []
        names = []
        for i in range(num_layers):
            layers.append(nn.Linear(in_features=layer_sequence[i],
                                    out_features=layer_sequence[i + 1],
                                    bias=False)
                         )
            names.append(f'fc{i + 1}')
                
            if add_relu and i != num_layers - 1:
                layers.append(nn.ReLU())
                names.append(f'relu{i + 1}')
        
        self.module_dict = OrderedDict(zip(names, layers))
        self.block = nn.Sequential(self.module_dict)
        
    def forward(self, x):
        out = self.block(x)
        
        return out

class NewsNet(nn.Module):
    
    def __init__(self, vocab, hidden_size=10, embedding_dim=16, num_layers=2, pretrained_embeddings=None):
        super(NewsNet, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.word_embeddings = nn.Embedding(num_embeddings=len(vocab), embedding_dim=self.embedding_dim)
        if pretrained_embeddings is not None:
            self.word_embeddings.weight.data.copy_(pretrained_embeddings.weight.data)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, 
                            bias=False, 
                            hidden_size=self.hidden_size, 
                            batch_first=True,
                            num_layers=self.num_layers)
        self.linear_block = LinearBlock([self.hidden_size, self.hidden_size * 2, self.hidden_size, 1])
        
    def forward(self, s):
        
        out = self.word_embeddings(s)
        sequence_out, (h, c) = self.lstm(out)
        out = h[-1]
        out = self.linear_block(out)
        
        return out