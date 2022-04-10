def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)


def predict(model, features):
  return Softmax(1)(model(features))
 

def train(EPOCHS = 20, BATCH = 32, trainingDataSet = trainingDataSet):
  criterion = CrossEntropyLoss()
  #Define SGD optimizer
  optimizer =  Adam(model.parameters()) #SGD(model.parameters(), lr=3e-4, momentum = 0.7, weight_decay=0.2)

  avg_train_loss_epoch = []
  train_acc_epoch = []

  # num_samples = 1000
  # sample_ds = torch.utils.data.Subset(trainingDataSet, np.arange(num_samples))
  # random_loader = torch.utils.data.RandomSampler(sample_ds)

  # trainLoader = torch.utils.data.DataLoader(sample_ds, sampler=random_loader, batch_size=BATCH)

  trainLoader = torch.utils.data.DataLoader(trainingDataSet, batch_size=BATCH)

  for epoch in range(EPOCHS):
      running_train_loss = 0 # 
      train_acc = 0      # training accuracy per batch

      for batchIndex, trainingData in tqdm(enumerate(trainLoader, 0),
                                           total=len(trainLoader)):

        #reset gradients
        optimizer.zero_grad()

        #forward-pass
        tInputs, tLabels = trainingData
        #print(model(tInputs))
        predictedY = model(tInputs)
        loss = criterion(predictedY, tLabels)
        loss.backward()

        if batchIndex == 0:
          plot_grad_flow(model.named_parameters())
          plt.show()

        running_train_loss += loss.item()*BATCH

        #update weights and reset gradient
        optimizer.step()

        train_prediction = predict(model, tInputs).argmax(1)

        train_acc += (train_prediction.flatten() == tLabels).sum()
      train_acc_epoch.append(train_acc/trainingDataSet.features.shape[0])
      avg_train_loss_epoch.append(running_train_loss/trainingDataSet.features.shape[0])

      print('Epoch %04d  Training Loss %f Training Accuracy %.2f' % (epoch + 1, running_train_loss/trainingDataSet.features.shape[0], 100*train_acc/trainingDataSet.features.shape[0]))
  
  #Plot training loss
  plt.title("Train Loss")
  plt.plot(avg_train_loss_epoch, label="Train")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.show()

  plt.title("Train Accuracy")
  plt.plot(train_acc_epoch, label="Train")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy (%)")
  plt.show()
