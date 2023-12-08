import torch



class HumanResponseModel(torch.nn.Module):
    """
    This is a nural network model to predict human response (valance and arousal) based on the HRC robot state
    """
    def __init__(self, input_size=5, hidden_size=32, output_size=2, dropout_rate=0):  # dropout too aggressive for online learning
        super(HumanResponseModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Define the layers
        self.linear1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.linear2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout2 = torch.nn.Dropout(dropout_rate)
        self.linear3 = torch.nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        x = torch.nn.functional.tanh(x) * 10
        return x