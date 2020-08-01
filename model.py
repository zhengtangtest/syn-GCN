import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
from torch_geometric.nn import GCNConv

SEED = 1234

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cpu')#"cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, syn_size, ner_size, hidden_size, pretrained):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding.from_pretrained(pretrained, freeze=False)
        # self.lemma_embedding = nn.Embedding(2, 5)
        self.ner_embedding = nn.Embedding(ner_size, 30)
        self.syn_embedding = nn.Embedding(syn_size, hidden_size)
        
        self.rnn = nn.LSTM(embedding_size + 30, hidden_size, bidirectional=True)

        self.linear  = nn.Linear(hidden_size * 2,   hidden_size)
        self.linear2 = nn.Linear(hidden_size + (embedding_size+30)*2, hidden_size)

        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, input, syn_labels, subj_pos, obj_pos, edge_index, ner):
        # lemma = [1 if i in subj_pos or i in obj_pos else 0 for i in range(input.size(0))]
        # lemma = torch.tensor(lemma, dtype=torch.long, device=device).view(-1, 1)
        # lemma_embeded = self.lemma_embedding(lemma).view(-1, 1, 5)

        ner_embeded = self.ner_embedding(ner).view(-1, 1, self.hidden_size)
        
        embedded = self.embedding(input).view(-1, 1, 300)
        embedded = torch.cat((embedded, ner_embeded), dim=2)
        
        syn_embedded         = self.syn_embedding(syn_labels).view(syn_labels.size(0), -1)
        first_word_embedded  = embedded[edge_index[0],:,:].view(syn_labels.size(0), -1)
        second_word_embedded = embedded[edge_index[1],:,:].view(syn_labels.size(0), -1)
        syn_embedded         = torch.cat((syn_embedded, first_word_embedded, second_word_embedded), 1)
        syn_embedded         = self.linear2(syn_embedded)

        output, hidden = self.rnn(embedded)
        output  = self.linear(output)
        outputs = output.view(input.size(0), -1)
        
        subj_vec = outputs[subj_pos[0]:subj_pos[-1]+1]
        obj_vec = outputs[obj_pos[0]:obj_pos[-1]+1]
        subj, sw = self.event_summary(subj_vec)
        obj, ow = self.event_summary(obj_vec)
        return outputs, subj, obj, sw, ow, syn_embedded

    def event_summary(self, event):
        attn_weights = F.softmax(torch.t(self.attn(event)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 event.unsqueeze(0))
        return attn_applied[0, 0], attn_weights

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Classifier, self).__init__()

        self.hidden_size = hidden_size

        self.attn = nn.Linear(input_size, hidden_size, bias=False)
        self.gcn = GCNConv(hidden_size, hidden_size)

        self.out = nn.Linear(hidden_size * 3, output_size)
        self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, encoder_outputs, syn_embeddeds, subj, obj, edge_index):
        edge_weights = F.softmax(
            torch.mm(
                self.attn(encoder_outputs[0].view( 1,-1)), torch.t(syn_embeddeds)
                )
            , dim=1)
        edge_weights = edge_weights.squeeze(0)
        outputs = self.gcn(encoder_outputs, edge_index, edge_weights)
        output = torch.cat((outputs[0], subj, obj))
        output = self.softmax(self.out(output))
        return output.unsqueeze(0)