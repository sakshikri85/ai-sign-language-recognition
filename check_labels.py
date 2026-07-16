
import numpy as np
import os

print("=== Model Labels ===")
labels = np.load("labels.npy", allow_pickle=True)
print(list(labels))

print("\n=== Dataset Folders ===")
folders = sorted(os.listdir("dataset"))
print(folders)