import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.signal import chirp, spectrogram

def plot_spectrogram(title, w, fs):
    ff, tt, Sxx = spectrogram(w, fs=fs, nperseg=int(fs / 2), nfft=len(w))
    plt.pcolormesh(tt, ff[:145], Sxx[:145], cmap='gray_r', shading='gouraud')
    plt.title(title)
    plt.xlabel('t (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.grid()

def create_chirps(freq_range=[0, 20], freq_removed=[5, 8], num_wins=1000, fs=50, window_sec=5, data_file='chirp_data_v0_0.pkl'):
    # Generating a dataset of chirp signals
    # Training and testing signals have ending frequencies between 0 and 20
    # Training signals have ending frequency of gyometric distribution with range of frequencies [0-20], and removing a range
    # of ending frequencies.
    # Testing signals have ending frequency of uniform distribution with range of frequencies [0-20].

    window_length = fs*window_sec
    #################generating the signals for training
    #The ending frequencies drawn from the exponential distribution (geometric)

    np.random.seed(200)
    exp_dist_freq = np.random.exponential(scale=2.1, size=num_wins)
    exp_dist_freq = np.delete(exp_dist_freq, np.logical_and(exp_dist_freq>freq_removed[0], exp_dist_freq<freq_removed[1]))
    num_wins = exp_dist_freq.shape[0]
    plt.figure()
    plt.hist(exp_dist_freq, bins=np.arange(0, np.max(exp_dist_freq+1)))
    plt.title('Training labels distribution')
    plt.savefig('./main_results/train_dist.png')
    plt.close()

    data_chirp_train = np.ndarray((num_wins, window_length, 1))
    win_info_train = np.ndarray((num_wins, 2)) # start frequency, end frequency
    t = np.linspace(0, window_length / fs, window_length)
    for i in range(num_wins):
        win_info_train[i, 0] = np.random.uniform(low=freq_range[0], high=freq_range[1], size=1) # start frequency
        win_info_train[i, 1] = exp_dist_freq[i] # end frequency
        data_chirp_train[i, :, 0] = chirp(t, f0=win_info_train[i, 0], f1=exp_dist_freq[i],
                  t1=window_length/fs, method='quadratic') #t1 is the time for the f1 frequency


    plt.figure(figsize=(20, 20))
    count1 = 1
    count2 = 2
    for i in range(0, 7, count1):
        plt.subplot(7, 2, count1)
        plt.plot(t, data_chirp_train[i,:])
        plt.title('Generated training Chirp, ending frequency=%2.2f' %
                  (win_info_train[i, 1]))
        plt.xlabel('t (sec)')
        plt.subplot(7, 2, count2)
        plot_spectrogram('Spectrogram of the chirps', data_chirp_train[i, :, 0], fs)
        #plot_periodogram('Power spectral density of the Chips, f=' + str(win_info[i,0]), data_chirp[i, :, 0], fs)
        plt.ylim([0, 20])
        count1 += 2
        count2 += 2
    plt.savefig('./main_results/generated_samples_train.png')
    plt.close()


    ####################generating the signals for testing
    #The ending frequencies drawn from the exponential distribution (geometric)
    uniform_dist_freq = np.random.uniform(freq_range[0], high=freq_range[1], size=num_wins)
    plt.figure()
    plt.hist(uniform_dist_freq, bins=np.arange(0, np.max(uniform_dist_freq+1)))
    plt.title('Testing labels distribution')
    plt.savefig('./main_results/test_dist.png')
    plt.close()

    data_chirp_test = np.ndarray((num_wins, window_length, 1))
    win_info_test = np.ndarray((num_wins, 2)) # start frequency, end frequency
    t = np.linspace(0, window_length / fs, window_length)
    for i in range(num_wins):
        win_info_test[i, 0] = np.random.uniform(low=freq_range[0], high=freq_range[1], size=1) # start frequency
        win_info_test[i, 1] = uniform_dist_freq[i] # end frequency
        data_chirp_test[i, :, 0] = chirp(t, f0=win_info_test[i, 0], f1=uniform_dist_freq[i],
                  t1=window_length/fs, method='quadratic') #t1 is the time for the f1 frequency

    plt.figure(figsize=(20, 20))
    count1 = 1
    count2 = 2
    for i in range(0, 7, count1):
        plt.subplot(7, 2, count1)
        plt.plot(t, data_chirp_test[i,:])
        plt.title('Generated testing Chirp, ending frequency=%2.2f' %
                  (win_info_test[i, 1]))
        plt.xlabel('t (sec)')
        plt.subplot(7, 2, count2)
        plot_spectrogram('Spectrogram of the chirps', data_chirp_test[i, :, 0], fs)
        #plot_periodogram('Power spectral density of the Chips, f=' + str(win_info[i,0]), data_chirp[i, :, 0], fs)
        plt.ylim([0, 20])
        count1 += 2
        count2 += 2
    plt.savefig('./main_results/generated_samples_test.png')
    plt.close()

    with open(data_file, 'wb') as data:  # Python 3: open(..., 'wb')
        pickle.dump([data_chirp_train, win_info_train, data_chirp_test, win_info_test, fs, window_length, freq_range], data)
        
    return data_chirp_train, win_info_train, data_chirp_test, win_info_test, fs, window_length, freq_range

if __name__ == '__main__':
    data_chirp_train, win_info_train, data_chirp_test, win_info_test, fs, window_length, freq_range = create_chirps()