import h5py
import os

hdf5_file = r"data/preprocessed/C1_001_PSG1.hdf5"

if not os.path.exists(hdf5_file):
    print(f"❌ FILE NOT FOUND: {hdf5_file}")
    print(f"\n📁 Contents of data/preprocessed/:")
    if os.path.exists("data/preprocessed"):
        for f in os.listdir("data/preprocessed"):
            print(f"  - {f}")
    else:
        print("❌ Directory data/preprocessed/ doesn't exist!")
else:
    print(f"✅ File exists: {hdf5_file}")
    file_size_mb = os.path.getsize(hdf5_file) / (1024 * 1024)
    print(f"📦 Size: {file_size_mb:.2f} MB")
    
    print(f"\n📊 Contents:")
    with h5py.File(hdf5_file, 'r') as f:
        if len(f.keys()) == 0:
            print("❌ FILE IS EMPTY (no datasets)")
        else:
            for key in f.keys():
                shape = f[key].shape
                dtype = f[key].dtype
                duration_min = shape[0] / 128 / 60
                print(f"  ✅ {key}: shape={shape}, dtype={dtype}, duration={duration_min:.1f} min")