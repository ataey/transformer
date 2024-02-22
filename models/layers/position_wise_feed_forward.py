from torch import nn

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, dropout_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, hidden)
        self.linear_2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

# if __name__ == '__main__':
#     return
