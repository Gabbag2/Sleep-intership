import os

N = 'N1:050'
H = 'HI:097'
C = 'C1:049'
label = 'PSG4_Hypnogram_Export_C1_014_PSG1'
all_sets = [N, H, C]

data_dir = "data"

# récupérer les fichiers présents
files_present = set(os.listdir(data_dir))

for a in all_sets:
    v1, v2 = a.split(":")
    for i in range(1, int(v2) + 1):
        test = f"{v1}_{i:03d}_PSG1.edf"
        
        if test not in files_present:
            print(test)
        
        test2 = 'PSG4_Hypnogram_Export_' + test.split("_PSG1.edf")[0]
        if test2 not in files_present:
            print(test2)