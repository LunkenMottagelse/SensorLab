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
fname = 'lab4Test_with_car_speed2_1.bin'
try:
    sample_period, data = raspi_import(fname)
except FileNotFoundError as err:
    print(f"File {fname} not found. Check the path and try again.")
    exit(1)


# Uncomment to remove linear in/decrease and DC component
#data = signal.detrend(data, axis=0)
# sample period is given in microseconds, so this changes units to seconds
sample_period *= 1e-6
data = data[10000:]

# Generate time axis
num_of_samples = data.shape[0]  # returns shape of matrix

t = np.linspace(0, num_of_samples*sample_period, num_of_samples,
        endpoint=False)


# Generate frequency axis and take FFT
# Use FFT shift to get monotonically increasing frequency
freq = np.fft.fftfreq(n=num_of_samples, d=sample_period)
freq = np.fft.fftshift(freq)
# takes FFT of all channels
spectrum = np.fft.fftshift(np.fft.fft(data, axis=0), axes=0)

data = data * 3.3 / 4096

data1 = []
data2 = []
for row in data:
    #ADC numbers are changed to other mic numbers:
    data1.append(row[3]) # ADC 4: Q signal
    data2.append(row[4]) # ADC 5: I signal

# Separerer data ut i egne lister (ikke optimalisert men funker på små datasett)

spectrum0 = []
spectrum1 = []
spectrum2 = []
spectrum3 = []
spectrum4 = []

for row in abs(spectrum):
    spectrum0.append(row[0])
    spectrum1.append(row[1])
    spectrum2.append(row[2])
    spectrum3.append(row[3])
    spectrum4.append(row[4])

zeros = np.zeros(len(spectrum0))

data1 = np.array(data1)
data2 = np.array(data2)

complexData = data2 + 1j * data1

complexSpectrum = np.fft.fftshift(np.fft.fft(complexData))

complexSpectrum[len(complexSpectrum)//2] = 1

# print(complexData)

topIndex = np.argmax(20*np.log10(np.abs(complexSpectrum)))
topFreq = freq[topIndex]
radarFreqPerC = 24.13 * 10 / 3

speed = topFreq / (radarFreqPerC * 2)
print("Maximum Frequency:", topFreq)
print("Vehicle speed:", speed)

i1 = np.where(freq <= 305)[0][-1]
i2 = np.where(freq <= 310)[0][-1]

bucket_305_less = complexSpectrum[:i1]
bucket_305_310 = complexSpectrum[i1:i2]
bucket_310_more = complexSpectrum[i2:]
outside_bucket = np.concatenate([bucket_305_less, bucket_310_more])o

snr = np.mean(np.abs(bucket_305_310))-np.mean(np.abs(outside_bucket))
snr = 10*np.log10(snr) # Wrong size but noise close to the signal is more relevant

print("SNR [dB]:", snr)
# Plot the results in two subplots
# If you want a single channel, use data[:,n] to get channel n
# For the report, write labels in the same language as the report
plt.subplot(2, 1, 1)
plt.title("Time domain signal")
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
#plt.xlim(0, 0.1)
plt.plot(t, data1)
plt.plot(t, data2)
plt.ylim(0, 3.3)
plt.grid()

plt.subplot(2, 1, 2)
plt.title("Power spectrum of signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power [dB]")
plt.minorticks_on()
plt.grid(True, which='major')
plt.grid(True, which='minor', linestyle=':')
# plt.xlim(0, 15620/2)
# Plot positive half of the spectrum (why?)
# plt.plot(freq[len(freq)//2:], 20*np.log10(np.abs(spectrum0[len(freq)//2:]))-np.amax(20*np.log10(np.abs(spectrum0[len(freq)//2:])))+6)
# plt.plot(freq[len(freq)//2:], 20*np.log10(np.abs(spectrum1[len(freq)//2:]))-np.amax(20*np.log10(np.abs(spectrum1[len(freq)//2:])))+6)
# plt.plot(freq[len(freq)//2:], 20*np.log10(np.abs(spectrum2[len(freq)//2:]))-np.amax(20*np.log10(np.abs(spectrum2[len(freq)//2:])))+6)
# plt.plot(freq, 20*np.log10(np.abs(spectrum3))-np.amax(20*np.log10(np.abs(spectrum3)))+6)
# plt.plot(freq, 20*np.log10(np.abs(spectrum4))-np.amax(20*np.log10(np.abs(spectrum4)))+6)
plt.plot(freq, 20*np.log10(np.abs(complexSpectrum))-np.amax(20*np.log10(np.abs(complexSpectrum))))
# Plotting zero line:
plt.plot(freq, zeros, color='k')
# Required if you have not called plt.ion() first
plt.show()
