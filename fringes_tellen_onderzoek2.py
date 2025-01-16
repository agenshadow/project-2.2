import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal


file: str= "MeetData/meting5/Raw Data.csv"

df = pd.read_csv(file)

array = df.to_numpy()
tijd = array[:,0]
illum = array[:,1]





beginseconden = 1
eindmin = float(input("hoeveelheid minuten: "))
eindseconden = float(input("hoeveelheid seconden: "))
begin = int(beginseconden/0.094524288)
eind = int((eindmin*60+eindseconden)/0.094524288)

tijd = tijd[begin:eind]
illum = illum[begin:eind]





peaks = signal.find_peaks(illum,distance=2.7/0.094524288,height=240)[0]

xpeaks = tijd[peaks]
ypeaks = illum[peaks]

print(len(xpeaks))


plt.plot(tijd, illum)
plt.scatter(xpeaks, ypeaks, c="orange")
plt.show()