undenoised_sequences = torch.as_tensor(np.array(all_sequences)).squeeze(3)
all_labels =  torch.as_tensor(np.array(all_labels))

def get_parent_class(index):
    if index < 5: #class N
      return 0
    elif index < 9:
      return 1 #class S
    elif index < 12:
      return 2 #class V
    elif index ==12:
      return 3 #class F
    else:
      return 4 #class Q
def get_counts(labels):
  class_counts = [0]*5
  # Count the number of labels in the training set that belong to each class
  for i, v in enumerate(labels):
      if v < 5:
        class_counts[0] += 1 #class N
      elif v < 9:
        class_counts[1] += 1 #class S
      elif v < 12:
        class_counts[2] += 1 #class V
      elif v ==12:
        class_counts[3] += 1 #class F
      else:
        class_counts[4] += 1 #class Q
  return class_counts



# Performs a 0.5 split because we will be adding more data into the training set later

noisy_train, noisy_test, noisy_train_labels, noisy_test_labels = train_test_split(np.array(undenoised_sequences), np.array(all_labels), test_size=0.5, random_state=42, stratify = np.array(all_labels))

class_counts = get_counts(noisy_train_labels)
max_count_index = np.argmax(class_counts) # Is most likely class N, but we'll do this anyway

labels, label_counts = np.unique(noisy_train_labels, return_counts=True)

indices_by_label = {labels[i]: [] for i in range(len(label_counts))}
for indx, label in enumerate(noisy_train_labels):
  indices_by_label[label.item()].append(indx)

final_indices = []
final_labels = []
for label, count in zip(labels, label_counts):
  # number of samples needed according to proportion 
  num_to_append = int(class_counts[max_count_index] * label_counts[label]/class_counts[get_parent_class(label)]) 
  final_indices.append(np.random.choice(indices_by_label[label], size = num_to_append))
  final_labels.append(np.repeat(label, num_to_append))

final_indices = np.concatenate(final_indices)
final_labels = np.concatenate(final_labels)

shuffle_indices = np.arange(len(final_indices))
np.random.shuffle(shuffle_indices)
noisy_final =noisy_train[final_indices[shuffle_indices]]
noisy_final_labels = final_labels[shuffle_indices]



# Convert the labels to their parent classes
for labels in (noisy_final_labels, noisy_test_labels):
  for i, v in enumerate(labels):
    labels[i] = get_parent_class(v)

noisy_final = torch.as_tensor(noisy_final)
noisy_final_labels = torch.as_tensor(noisy_final_labels) 

noisy_test = torch.as_tensor(noisy_test)
noisy_test_labels = torch.as_tensor(noisy_test_labels) 



# Do not oversample or generate synth data for testing
noisytrainingFeatures, noisytrainingLabels = noisy_final, noisy_final_labels
noisytestFeatures, noisytestLabels = noisy_test, noisy_test_labels
