# declare model1 if want to train it
model = CNN()

#declare model2 if need to train model2
model = LSTM_CNN()

#train with denoising
train()

#train without denoising
train(20, 32, noisytrainingDataSet)

# test with denoising
test_prediction = predict(model, testDataSet.features).argmax(1)
test_acc = (test_prediction.flatten() == testDataSet.labels).sum()/testDataSet.features.shape[0]
test_acc 

# test without denoising
test_prediction = predict(model, testDataSet.features).argmax(1)
test_acc = (test_prediction.flatten() == testDataSet.labels).sum()/testDataSet.features.shape[0]
test_acc 
