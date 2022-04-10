def denoise_signal(ecg_signal):
  # Perform daubhechies-6 wavelet decomposition
  coeffs = pywt.wavedec(ecg_signal, 'db6', mode='per')

  # Get mean absolute deviation of decomposed signal
  mean_abs_dev = np.mean(np.abs(coeffs[-1] - np.mean(coeffs[-1])))
  # Set threshold for wavelet damping
  threshold_const = 1.1
  uthresh = threshold_const*mean_abs_dev * np.sqrt(2 * np.log(len(ecg_signal)))
  
  coeffs[1:] = [pywt.threshold(coeff_val, value=uthresh, mode='hard') for
                coeff_val in coeffs[1:]]
  return pywt.waverec(coeffs, 'db6', mode='per')


all_sequences = torch.as_tensor(np.array(all_sequences)) # runtime for this cell went from ~3.5min to ~0sec. by changing method of converting all_* to a torch tensor

# Must denoise data before inputting into the CNN 

denoised_intermediate = torch.stack([
                    torch.from_numpy(denoise_signal(x_i.squeeze())) for i, x_i in enumerate(torch.unbind(all_sequences, dim=0), 0) ], dim=0)
denoised_sequences = denoised_intermediate[:, None, :]


# Performs a 0.5 split because we will be adding more data into the training set later

train_sequences, test_sequences, train_labels, test_labels = train_test_split(np.array(denoised_sequences), np.array(all_labels), test_size=0.5, random_state=42, stratify = np.array(all_labels))

class_counts = get_counts(train_labels)

max_count_index = np.argmax(class_counts) # Is most likely class N, but we'll do this anyway

labels, label_counts = np.unique(train_labels, return_counts=True)

indices_by_label = {labels[i]: [] for i in range(len(label_counts))}
for indx, label in enumerate(train_labels):
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
final_sequences = train_sequences[final_indices[shuffle_indices]]
final_labels = final_labels[shuffle_indices]


# Convert the labels to their parent classes
for labels in (final_labels, test_labels):
  for i, v in enumerate(labels):
    labels[i] = get_parent_class(v)

final_sequences = torch.as_tensor(final_sequences)
final_labels = torch.as_tensor(final_labels) 

test_sequences = torch.as_tensor(test_sequences)
test_labels = torch.as_tensor(test_labels) 


# Do not oversample or generate synth data for testing
trainingFeatures, trainingLabels = final_sequences, final_labels
testFeatures, testLabels = test_sequences, test_labels

AllFeatures, AllLabels = torch.cat((final_sequences, test_sequences)), torch.cat((final_labels, test_labels)) 
