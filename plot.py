import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Excel file
df = pd.read_csv("E:/Jason_W/Documents/Projects/Blackbody-Simulations/HFSSSimData/InfParallelPlate_bbsim3_500GHz_Ephi=0/waveguide.csv")
df2 = pd.read_csv("E:/Jason_W/Documents/Projects/Blackbody-Simulations/HFSSSimData/InfParallelPlate_bbsim3_500GHz_Ephi=1/waveguide.csv")

# Drop duplicates (optional, safe if OutgoingPower is always same)
df_unique = df.drop_duplicates(subset=["IWavePhi", "IWaveTheta"])
df2_unique = df2.drop_duplicates(subset=["IWavePhi", "IWaveTheta"])

# Pivot table for heatmap: rows = IWaveTheta, columns = IWavePhi
pivot = df_unique.pivot(index="IWaveTheta", columns="IWavePhi", values="OutgoingPower")
pivot2 = df2_unique.pivot(index="IWaveTheta", columns="IWavePhi", values="OutgoingPower")

# Plotting
plt.figure(figsize=(8, 6))
sns.heatmap(pivot, annot=False, fmt=".2e", cmap="viridis")
plt.title("Outgoing Power vs IWaveTheta and IWavePhi (E_phi=0)")
plt.xlabel("Incoming Wave Phi")
plt.ylabel("Incoming Wave Theta")
plt.tight_layout()

plt.figure(figsize=(8, 6))
sns.heatmap(pivot2, annot=False, fmt=".2e", cmap="viridis")
plt.title("Outgoing Power vs IWaveTheta and IWavePhi (E_phi=1)")
plt.xlabel("Incoming Wave Phi")
plt.ylabel("Incoming Wave Theta")
plt.tight_layout()
plt.show()