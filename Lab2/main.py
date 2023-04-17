import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def raspi_import(path, channels=5):
    """
    Import data produced using adc_sampler.c.
    Returns sample period and ndarray with one column per channel.
    Sampled data for each channel, in dimensions NUM_SAMPLES x NUM_CHANNELS.
    """

    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype='uint16').astype('float64')
        # The "dangling" `.astype('float64')` casts data to double precision
        # Stops noisy autocorrelation due to overflow
        data = data.reshape((-1, channels))
    return sample_period, data


# Import data from bin file
fname = 'sampleData/officialTest2_-130_sample7.bin'
try:
    sample_period, data = raspi_import(fname)
except FileNotFoundError as err:
    print(f"File {fname} not found. Check the path and try again.")
    exit(1)

# Uncomment to remove linear in/decrease and DC component
data = signal.detrend(data, axis=0)
# sample period is given in microseconds, so this changes units to seconds
sample_period *= 1e-6

# Generate time axis
num_of_samples = data.shape[0]  # returns shape of matrix

t = np.linspace(0, num_of_samples*sample_period, num_of_samples,
        endpoint=False)

data1 = []
data2 = []
data3 = []
for row in data:
    #ADC numbers are changed to other mic numbers:
    data1.append(row[2]) # ADC 3 -> Mic 1
    data2.append(row[0]) # ADC 1 -> Mic 2
    data3.append(row[1]) # ADC 2 -> Mic 2

#plt.plot(data1)
#plt.show()

# Cutting off early samples to remove noise from the early signals

cutoff_amount = 1500

data1 = data1[cutoff_amount:]
data2 = data2[cutoff_amount:]
data3 = data3[cutoff_amount:]

num_of_samples -= cutoff_amount


#Upsampling the data
upsamplingFactor = 3
num_of_samples *= upsamplingFactor
upsampled1 = signal.resample(data1, num_of_samples)
upsampled2 = signal.resample(data2, num_of_samples)
upsampled3 = signal.resample(data3, num_of_samples)

#plt.plot(upsampled1)
#plt.show()


cc_21 = signal.correlate(upsampled2, upsampled1, "full")
abs_cc_21 = np.abs(cc_21)
n_21 = np.argmax(abs_cc_21)
n_21 -= num_of_samples

cc_31 = signal.correlate(upsampled3, upsampled1, "full")
abs_cc_31 = np.abs(cc_31)
n_31 = np.argmax(abs_cc_31)
n_31 -= num_of_samples

cc_32 = signal.correlate(upsampled3, upsampled2, "full")
abs_cc_32 = np.abs(cc_32)
n_32 = np.argmax(abs_cc_32)
n_32 -= num_of_samples

print("n_21:", n_21, "\t n_31:", n_31, "\t n_32:", n_32)

angle = np.arctan2(np.sqrt(3) * (n_21 + n_31), (n_21 - n_31 - (2* n_32)))

angle = angle * 180 / np.pi

print("Angle:", angle)

printable = abs_cc_21[num_of_samples:num_of_samples + 100]

#plt.plot(abs_cc_21)
#plt.show()