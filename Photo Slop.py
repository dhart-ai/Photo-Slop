import numpy as np
import urllib.request
import matplotlib.pyplot as plt

categories = ["cat", "dog", "horse", "cow"]

# Step 1: download each animal's doodle file
for c in categories:
    url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{c}.npy"
    urllib.request.urlretrieve(url, f"{c}.npy")

# Step 2: load each file and print how many doodles it holds
datasets = {}
for c in categories:
    datasets[c] = np.load(f"{c}.npy")
    print(c, datasets[c].shape)

# Step 3: show the first 6 doodles of each animal and save the picture
fig, axes = plt.subplots(len(categories), 6, figsize=(9, 6))
for row, c in enumerate(categories):
    for col in range(6):
        axes[row, col].imshow(datasets[c][col].reshape(28, 28), cmap="gray_r")
        axes[row, col].axis("off")
    axes[row, 0].set_title(c, loc="left")
plt.tight_layout()
plt.savefig("sample_doodles.png", dpi=150)
plt.show()