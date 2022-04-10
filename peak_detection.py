import scipy.signal

def detect_peaks(ecg_signal, sampling_rate):
  # Use Pan-Tompkins algorithm to detect R-peaks
  W1     = 5*2/sampling_rate                                    
  W2     = 15*2/sampling_rate                        
  b, a   = scipy.signal.butter(4, [W1,W2], 'bandpass')                  
  ecg_signal = scipy.signal.filtfilt(b,a, ecg_signal) 

  diff = - ecg_signal[:-2] - 2 * ecg_signal[1:-1] + 2 * ecg_signal[1:-1] + ecg_signal[2:]
  diff_squared = np.square(diff)

  window_size = int(0.12*sampling_rate)

  moving_avg = scipy.ndimage.uniform_filter1d(diff_squared,
                                              window_size,
                                              origin=(window_size-1)//2)

  diff_winsize = int(0.2*sampling_rate)
  moving_avg_diff = scipy.signal.convolve(moving_avg, np.ones(diff_winsize, ), mode='same')

  peaks = []
  for i in range(1, len(moving_avg_diff) - 1):
    h0, h, h1 = moving_avg_diff[i-1], moving_avg_diff[i], moving_avg_diff[i+1]

    if h > h0 and h > h1:
      peaks.append(i)

  prob_rpeaks = []
  for approx_rpeak in peaks:
    win = ecg_signal[max(0, approx_rpeak-(window_size)):
                     min(len(ecg_signal), approx_rpeak+(window_size))]
    peak_val = max(win)

    prob_rpeak = approx_rpeak-(window_size) + np.argmax(win)
    prob_rpeaks.append(prob_rpeak)

  rpeaks = []

  noise_level = 0
  signal_level = 0
  for i, peak in enumerate(prob_rpeaks):
    thresh = noise_level + 0.25 * (signal_level - noise_level)

    peak_height = moving_avg[peak]

    if peak_height > thresh:
      signal_level = 0.125 * peak_height + 0.875 * signal_level
      rpeaks.append(peak)
    else:
      noise_level = 0.125 * peak_height + 0.875 * noise_level

  filter_peaks = [rpeaks[0]]
  refractory_period = 0.2*sampling_rate
  twave_window_thresh = 0.3*sampling_rate

  for i in range(1, len(rpeaks)):
    if (rpeaks[i] - rpeaks[i - 1] < twave_window_thresh and
        rpeaks[i] - rpeaks[i - 1] > refractory_period):
      if moving_avg[rpeaks[i]] > (moving_avg[rpeaks[i - 1]] / 2):
        filter_peaks.append(rpeaks[i])
    else:
      filter_peaks.append(rpeaks[i])

  rpeaks = filter_peaks
  
  fig = plt.figure(figsize=(24, 3))
  plt.plot(ecg_signal)
  for rpeak in rpeaks:
    plt.plot(rpeak, ecg_signal[rpeak], 'ro')
  for peak in rpeaks:
    plt.axvspan(peak-window_size, peak+window_size, alpha=0.3)
  plt.show()
  # plt.plot(moving_avg)
  # for peak in peaks:
  #   plt.plot(peak, moving_avg[peak], 'go')
  # plt.show()

  return rpeaks
#all_sequences[8661][0].squeeze().shape
for i in range(5):
  detect_peaks(np.concatenate([
      all_sequences[j][0] for j in range(10*i, 10*(i+1))
  ]).squeeze(), 360)
  
  
  
  # TESTING R PEAK DETECTION

start, end = 3000, 12000
subject = 119
record = wfdb.rdrecord(f'mitdb/{subject}')
annotation = wfdb.rdann(f'mitdb/{subject}', 'atr')
atr_symbol = annotation.symbol
atr_sample = annotation.sample
rpeaks = detect_peaks(record.p_signal[start:end,0], 360)
peaks = np.array(rpeaks)

fig = plt.figure(figsize=(30, 5))
for peak in peaks:
  plt.plot(peak, record.p_signal[start+peak, 0], 'ro', ms=15)
for indx, sample in enumerate(atr_sample):
  if sample > start and sample < end:
    plt.plot(sample - start, record.p_signal[sample, 0], 'go', ms=8)
    plt.text(sample - start, record.p_signal[sample, 0], atr_symbol[i], size=20)
plt.plot(record.p_signal[start:end,0])
plt.show()


squeezed_sequences = []
for signal in range(len(all_sequences)):
  squeezed_sequences.append(all_sequences[signal].squeeze())
concat_sequences = np.concatenate(squeezed_sequences)
peaks = detect_peaks(concat_sequences, 360)
len(peaks)
