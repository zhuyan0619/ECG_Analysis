from sklearn.metrics import confusion_matrix

#EDITS: Replaced TrainingDataSet with CrossValidationDataset, since the 10 folds are created using the entire dataset. 

#Initialize NN and list of trained models and loss function

criterion = CrossEntropyLoss()
k_folds = 10
kfold = KFold(n_splits=k_folds, shuffle=True) #(ten-fold cross-validation, 9/10 training 1/10 testing)

#Iterates for num of folds
class_acc = []
class_sens = []
class_ppv = []
class_spec = []
for fold, (train_ids, test_ids) in enumerate(kfold.split(CrossValidationDataSet)): 
  
  # Sample elements randomly from a given list of ids, no replacement.
  trainSubsampler = torch.utils.data.SubsetRandomSampler(train_ids)
  #testSubsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
  # Define data loaders for training and testing data in this fold
  trainLoader = torch.utils.data.DataLoader(CrossValidationDataSet, batch_size=10, sampler=trainSubsampler)
  #testLoader = torch.utils.data.DataLoader(CrossValidationDataSet, batch_size=256, sampler=testSubsampler)

  #Initialize CNN
  model = CNN()
  model.to(device)
  #Need to reset weights every fold
  
  #Define optimizer
  optimizer = Adam(model.parameters())

  #Train CNN
  epochs = 20
  for epoch in range(epochs):
    TotalLoss = 0
    #print("Epoch", epoch+1)
    for batchIndex, trainingData in enumerate(trainLoader, 0):

      #reset gradients
      optimizer.zero_grad()

      #forward-pass
      tInputs, tLabels = trainingData
      tInputs, tLabels = tInputs.to(device), tLabels.to(device)
      predictedY = model(tInputs)
      loss = criterion(predictedY, tLabels)

      #backward-pass
      loss.backward()

      #update weights and reset gradient
      optimizer.step()
      optimizer.zero_grad()

      #Display performance
      TotalLoss += loss.item()
      if (batchIndex+1) % 2000 == 0:    # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' %(epoch + 1, batchIndex + 1, TotalLoss / 2000))
        TotalLoss = 0.0
  
  # Test on test set, report accuracy, specificity and sensitivity
  test_prediction = predict(model, CrossValidationDataSet.features[test_ids]).argmax(1)
  conf = confusion_matrix(CrossValidationDataSet.labels[test_ids], test_prediction)
  acc = []
  sens = []
  ppv = []
  spec = []
  for i in range(5):
    tp = conf[i,i] 
    indices = np.append(np.arange(i), np.arange(i+1, 5)) # all except ith index
    tn = conf[indices, indices].sum()
    fp = conf[i, indices].sum()
    fn = conf[indices, i].sum()
    acc.append((tp + tn)/ (tp + tn + fp + fn))
    sens.append(tp/(tp + fn))
    ppv.append(tp/(tp + fp))
    spec.append(tn/(tn + fp))
  class_acc.append(np.array(acc))
  class_sens.append(np.array(sens))
  class_ppv.append(np.array(ppv))
  class_spec.append(np.array(spec))

  print("Training for fold %d has been completed, saving model" %(fold))

  # Saving the model
  save_path = f'./model-fold-{fold}.pth'
  torch.save(model.state_dict(), save_path)

# Get average values for model performance measures
avg_acc = np.array(class_acc).mean(0)
avg_sens = np.array(class_sens).mean(0)
avg_ppv = np.array(class_ppv).mean(0)
avg_spec = np.array(class_spec).mean(0)
