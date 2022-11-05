import torch
import torch.nn as nn
from hello.dataWorld.audioTools import get_spectr

class SpRecurrent(nn.Module):
    def __init__(self,input_size=128,num_layers=1,hidden_size=256,embedding_size=128,num_classes=2,bidir=False,dropout=0.5,device="cpu",model_type="GRU"):
        super(SpRecurrent, self).__init__()
        self.input_size=input_size
        self.num_layers=num_layers
        self.hidden_size=hidden_size
        self.embedding_size=embedding_size
        self.num_classes=num_classes
        self.dropout=dropout
        self.bidir=bidir
        self.device=device
        self.model_type=model_type

        if self.model_type=="GRU":
            self.model=nn.GRU(input_size=self.input_size,num_layers=self.num_layers,
                            hidden_size=self.hidden_size,dropout=self.dropout,batch_first=True,bidirectional=self.bidir)
        elif self.model_type=="LSTM":
            self.model=nn.LSTM(input_size=self.input_size,num_layers=self.num_layers,
                            hidden_size=self.hidden_size,dropout=self.dropout,batch_first=True,bidirectional=self.bidir)

        self.lstm_relu = nn.ReLU()
        
        if self.bidir==False:
            self.fc_1 =  nn.Linear(self.hidden_size, self.embedding_size) #fully connected 1
        elif self.bidir==True:
            self.fc_1 =  nn.Linear(self.hidden_size*2, self.embedding_size) #fully connected 1

        
        self.fc_1_relu = nn.ReLU()
        
        self.fc_2 = nn.Linear(self.embedding_size, self.num_classes) #fully connected last layer
        self.fc_2_softmax = nn.Softmax(dim=1)

       
       
      
        
    def forward(self, x):
        #Layer,Batch,hidden NOTE: random initialization could provide different output from same input
        h0,c0=None,None
        if self.bidir==False:
            h0,c0=(torch.randn(self.num_layers,x.size(0),self.hidden_size).to(self.device),torch.randn(self.num_layers,x.size(0), self.hidden_size).to(self.device)) # clean out hidden state
        elif self.bidir==True:
            h0,c0=(torch.randn(2*self.num_layers,x.size(0),self.hidden_size).to(self.device),torch.randn(2*self.num_layers,x.size(0), self.hidden_size).to(self.device)) # clean out hidden state

            
        #Output is the last layer for each time step [BS,seq,features] hn,cn is the last output (time step) for each layer
        if self.model_type=="LSTM":
            output, (hn, cn) = self.model(x, (h0,c0))
        elif self.model_type=="GRU":
            output, hn = self.model(x, h0)
        out=torch.reshape(output,(output.size(0),output.size(2),output.size(1)))
        out=out[:,:,-1] #Get only last timestep of last layer
        #out = output.view(-1, self.hidden_size)[-1] #reshaping the data for Dense layer next and get only last 
        out = self.lstm_relu(out)
        
        out = self.fc_1(out) #first Dense
        out = self.fc_1_relu(out)
        out = self.fc_2(out) #Final Output
        out = self.fc_2_softmax(out)
       
        return out

    
def recurrent_to_spectr_prepro(s):
    sig,label=s
    x=torch.tensor(get_spectr(sig,n_fft=8800))
    return torch.reshape(x,(x.shape[1],x.shape[0])),label