import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
import sys


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

filename = sys.argv[1]
savename = sys.argv[2]
f = open(filename, "r")
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

rFiltered = butter_highpass_filter(rFiltered, cutoffHigh, fs, order)
gFiltered = butter_highpass_filter(gFiltered, cutoffHigh, fs, order)
bFiltered = butter_highpass_filter(bFiltered, cutoffHigh, fs, order)


# #rCorr = signal.correlate(rNpDET, rNpDET, "full")
# rCorr = signal.correlate(rFiltered, rFiltered, "full")
# rCorr = np.abs(rCorr)
# #gCorr = signal.correlate(gNpDET, gNpDET, "full")
# gCorr = signal.correlate(gFiltered, gFiltered, "full")
# gCorr = np.abs(gCorr)
# #bCorr = signal.correlate(bNpDET, bNpDET, "full")
# bCorr = signal.correlate(bFiltered, bFiltered, "full")
# bCorr = np.abs(bCorr)
# corrLen = len(rCorr)

# rCorr = np.copy(rCorr[corrLen//2:corrLen//2 + 100])
# gCorr = np.copy(gCorr[corrLen//2:corrLen//2 + 100])
# bCorr = np.copy(bCorr[corrLen//2:corrLen//2 + 100])

# disregarded = 12
# rDelay = np.argmax(rCorr[disregarded:]) + disregarded
# gDelay = np.argmax(gCorr[disregarded:]) + disregarded
# bDelay = np.argmax(bCorr[disregarded:]) + disregarded

# rPulse = 60 * 40 / rDelay
# gPulse = 60 * 40 / gDelay
# bPulse = 60 * 40 / bDelay

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


plt.figure(num=(filename+": figure 1"))

plt.plot(freq, gFFT, color="g")
plt.plot(freq, rFFT, color="r")
plt.plot(freq, bFFT, color="b")
plt.xlim(0, 4)
plt.grid()
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power [mW]")
plt.savefig(savename+"_fft.pdf", format='pdf', bbox_inches='tight')



i1 = np.where(freq <= 1)[0][-1]
i2 = np.where(freq <= 3)[0][-1]

rBucket0_1 = rFFT[:i1]
rBucket1_3 = rFFT[i1:i2]
rBucket3_20 = rFFT[i2:]

gBucket0_1 = gFFT[:i1]
gBucket1_3 = gFFT[i1:i2]
gBucket3_20 = gFFT[i2:]

bBucket0_1 = bFFT[:i1]
bBucket1_3 = bFFT[i1:i2]
bBucket3_20 = bFFT[i2:]

rNoise = (np.mean(rBucket3_20) + np.mean(rBucket0_1)) / 2
gNoise = (np.mean(gBucket3_20) + np.mean(gBucket0_1)) / 2
bNoise = (np.mean(bBucket3_20) + np.mean(bBucket0_1)) / 2

rSNR = 10 * np.log10(np.mean(rBucket1_3)/rNoise)
gSNR = 10 * np.log10(np.mean(gBucket1_3)/gNoise)
bSNR = 10 * np.log10(np.mean(bBucket1_3)/bNoise)

rFreqMax = freq[np.argmax(rFFT)]
gFreqMax = freq[np.argmax(gFFT)]
bFreqMax = freq[np.argmax(bFFT)]

print("Heart rate using FFT:")
print(rFreqMax*60)
print(gFreqMax*60)
print(bFreqMax*60)

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

# print()
# print("Max lags:")
# print(rDelay)
# print(gDelay)
# print(bDelay)

# print("\nPulse (r/g/b)")
# print(rPulse)
# print(gPulse)
# print(bPulse)

# plt.figure(num=(filename+": figure 2"))
# plt.stem(gCorr, linefmt="green")
# plt.stem(bCorr, linefmt="blue")
# plt.stem(rCorr, linefmt="red")
# plt.xlabel("lags")
# plt.ylabel("correlation")
# plt.title("Color channel autocorrelation")
# plt.savefig(savename+"_corr.pdf", format='pdf', bbox_inches='tight')


print("Results:", rSNR, gSNR, bSNR, rFreqMax * 60, gFreqMax * 60, bFreqMax * 60)

input = input("Press any key . . . ")