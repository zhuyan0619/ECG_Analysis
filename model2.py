class LSTM_CNN(Module):
  """This class contains the CNN and LSTM model."""
  def __init__(self):
      super(LSTM_CNN, self).__init__()

      self.cnn_layers = Sequential(

          # Defining a 1D convolution layer (1 channel in, 32 out)
          Conv1d(1,32, kernel_size=3, stride=1),
          LeakyReLU(inplace=True),
          MaxPool1d(kernel_size=2, stride=1),
          
          # Defining another 1D convolution layer (32 channels in, 64 out)
          Conv1d(32, 64, kernel_size=4, stride=1),
          LeakyReLU(inplace=True),
          MaxPool1d(kernel_size=2, stride=1),

          # Defining another 1D convolution layer (64 channels in, 32 out)
          Conv1d(64, 32, kernel_size=4, stride=1),
          LeakyReLU(inplace=True),
          MaxPool1d(kernel_size=2, stride=1),
      )

      self.flatten = Flatten()

      #self.LSTM_layers = sequential()
      # self.LSTM_layers = LSTM(input=30*32,hidden_size=10,num_layers=1)
      self.LSTM_layers = torch.nn.LSTM(input_size=249,hidden_size=10,num_layers=1)   

      self.linear_layers = Sequential(
          #Linear layer (64 channels in, 128 out)
          Linear(320, 128), 
          LeakyReLU(inplace=True),

          #Linear layer with 5 neurons
          Linear(128, 5),
      )
    
  def forward(self, features):
    features = self.cnn_layers(features)
    # features = self.flatten(features)
    features, _ = self.LSTM_layers(features)
    features = self.flatten(features)
    features = self.linear_layers(features)
    return features
