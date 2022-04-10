class ECGDataset(Dataset):

    def __init__(self, features, labels):
     self.features = features.float() # input to torch.nn.Conv1d should be of shape (batch size, # channels, #length of sequence) ; use .squeeze(3) to get this ; also CNN needs input of type float
     self.labels = labels
     self.n_samples = labels.shape[0]

    def __getitem__(self, index):
      return self.features[index], self.labels[index]
    def __len__(self):
      return self.n_samples
  
trainingDataSet = ECGDataset(trainingFeatures, trainingLabels)
testDataSet = ECGDataset(testFeatures, testLabels)

noisytrainingDataSet = ECGDataset(noisytrainingFeatures, noisytrainingLabels)
noisytestDataSet = ECGDataset(noisytestFeatures, noisytestLabels)

CrossValidationDataSet = ECGDataset(AllFeatures, AllLabels)
