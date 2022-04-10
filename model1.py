class CNN(Module):   
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn_layers = Sequential(

            # Defining a 1D convolution layer
            Conv1d(1,5, kernel_size=3, stride=1), # kernel size = 3 in paper
            
            
            LeakyReLU(),
            MaxPool1d(kernel_size=2, stride=2),
            
            # Defining another 1D convolution layer
            Conv1d(5, 10, kernel_size=4, stride=1),

            LeakyReLU(),
            MaxPool1d(kernel_size=2, stride=2),

            #And another
            Conv1d(10, 20, kernel_size=4, stride=1), # 10, 20 in paper
            
            LeakyReLU(),
            MaxPool1d(kernel_size=2, stride=2)
            )

        self.flatten = Flatten()

        self.linear_layers = Sequential(
            #Linear layer with 30 neurons
            Linear(20*30,30),  
            LeakyReLU(),

            #Linear layer with 20 neurons
            Linear(30, 20), 
            LeakyReLU(),

            #Linear layer with 5 neurons
            Linear(20, 5), 

        )
    
    def forward(self, features):
        features = self.cnn_layers(features)
        features = self.flatten(features)
        features = self.linear_layers(features)
        return features
      
      
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #change colab runtime to GPU
from torchsummary import summary
test_model = CNN()
test_model.to(device)
summary(test_model, input_size = (1, 260))
