import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

r = []
g = []
b = []
fs = 40
cutoffLow = 3.5
cutoffHigh = 0.5
order = 5
f = open("Optikklab/Data/transmittance_7.txt", "r")
for line in f:
    l = line.split(" ")
    r.append(float(l[0]))
    g.append(float(l[1]))
    b.append(float(l[2]))


rNp = np.array(r)
gNp = np.array(g)
bNp = np.array(b)

rNpDET = signal.detrend(rNp)
gNpDET = signal.detrend(gNp)
bNpDET = signal.detrend(bNp)

rFiltered = butter_lowpass_filter(rNpDET, cutoffLow, fs, order)
gFiltered = butter_lowpass_filter(gNpDET, cutoffLow, fs, order)
bFiltered = butter_lowpass_filter(bNpDET, cutoffLow, fs, order)

#rFiltered = butter_highpass_filter(rFiltered, cutoffHigh, fs, order)
#gFiltered = butter_highpass_filter(gFiltered, cutoffHigh, fs, order)
#bFiltered = butter_highpass_filter(bFiltered, cutoffHigh, fs, order)


#rCorr = signal.correlate(rNpDET, rNpDET, "full")
rCorr = signal.correlate(rFiltered, rFiltered, "full")
rCorr = np.abs(rCorr)
#gCorr = signal.correlate(gNpDET, gNpDET, "full")
gCorr = signal.correlate(gFiltered, gFiltered, "full")
gCorr = np.abs(gCorr)
#bCorr = signal.correlate(bNpDET, bNpDET, "full")
bCorr = signal.correlate(bFiltered, bFiltered, "full")
bCorr = np.abs(bCorr)
corrLen = len(rCorr)

rCorr = np.copy(rCorr[corrLen//2:corrLen//2 + 100])
gCorr = np.copy(gCorr[corrLen//2:corrLen//2 + 100])
bCorr = np.copy(bCorr[corrLen//2:corrLen//2 + 100])

disregarded = 12
rDelay = np.argmax(rCorr[disregarded:]) + disregarded
gDelay = np.argmax(gCorr[disregarded:]) + disregarded
bDelay = np.argmax(bCorr[disregarded:]) + disregarded

rPulse = 60 * 40 / rDelay
gPulse = 60 * 40 / gDelay
bPulse = 60 * 40 / bDelay

rStd = np.std(rNp)
gStd = np.std(gNp)
bStd = np.std(bNp)

rMean = np.mean(rNp)
gMean = np.mean(gNp)
bMean = np.mean(bNp)

rFFT = np.abs(np.fft.fft(rFiltered, len(rFiltered) * 2))
gFFT = np.abs(np.fft.fft(gFiltered, len(rFiltered) * 2))
bFFT = np.abs(np.fft.fft(bFiltered, len(rFiltered) * 2))

n = rFiltered.size
timestep = 1 / fs
freq = np.fft.fftfreq(n * 2, d=timestep) 

fftLen = len(rFFT)

rFFT = np.copy(rFFT[:fftLen//2])
gFFT = np.copy(gFFT[:fftLen//2])
bFFT = np.copy(bFFT[:fftLen//2])
freq = np.copy(freq[:fftLen//2])



plt.plot(freq, gFFT, color="g")
plt.plot(freq, rFFT, color="r")
plt.plot(freq, bFFT, color="b")
plt.show()



i1 = np.where(freq <= 1)[0][-1]
i2 = np.where(freq <= 3)[0][-1]

rBucket0_1 = rFFT[:i1]
rBucket1_3 = rFFT[i1:i2]
rBucket3_20 = rFFT[i2:]


snr =  np.mean(rBucket1_3)-np.mean(rBucket3_20)

print("SNR: " + str(snr))

print()



print("Mean values:")
print(rMean)
print(gMean)
print(bMean)
print()
print("Standard devitaions")
print(rStd)
print(gStd)
print(bStd)

print()
print("Max lags:")
print(rDelay)
print(gDelay)
print(bDelay)

print("\nPulse (r/g/b)")
print(rPulse)
print(gPulse)
print(bPulse)

plt.stem(gCorr, linefmt="green")
plt.stem(bCorr, linefmt="blue")
plt.stem(rCorr, linefmt="red")
plt.show()
