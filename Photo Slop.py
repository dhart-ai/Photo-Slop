# --- Imports: tools the script needs ---
import ast                            # safely reads the file's header text as a Python dictionary
import urllib.request                 # downloads files from a web address
import numpy as np                    # works with big grids of numbers (our doodles)
import matplotlib.pyplot as plt       # draws pictures and charts
from sklearn.model_selection import train_test_split   # splits data into train/test
from sklearn.metrics import confusion_matrix           # counts which animals get mixed up

# --- Settings you can change ---
categories = ["cat", "dog", "horse", "cow"]   # the animals to download (lowercase, must match Quick, Draw! names)
n = 2000                                       # how many doodles to keep per animal

def fetch_doodles(name, n):
    """Download only the first n doodles of one animal.
    Returns the doodles and the total number the full file contains."""

    # Web address of this animal's file in the Quick, Draw! dataset
    url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{name}.npy"

    # Ask the server for just the start of the file instead of all ~100 MB.
    # Each doodle is 784 bytes, plus ~200 bytes for the file's header at the beginning.
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{200 + n * 784}"})
    buf = urllib.request.urlopen(req).read()    # the raw bytes we received

    # The file starts with a header describing its contents (like the total count).
    # Bytes 8-9 say how long that header is.
    hlen = int.from_bytes(buf[8:10], "little")

    # Read the header text as a dictionary; it includes the file's full shape.
    header = ast.literal_eval(buf[10:10 + hlen].decode("latin1"))
    total = header["shape"][0]                  # first number = total doodles in the full file

    # The doodle data begins right after the header
    start = 10 + hlen

    # Turn the raw bytes into a grid: one row per doodle, 784 pixel values per row (28 x 28 = 784)
    rows = np.frombuffer(buf[start:start + n * 784], dtype=np.uint8).reshape(-1, 784)
    return rows, total

# --- Step 1: download each animal and print how much we got ---
datasets = {}                                   # will hold the doodles for each animal
for c in categories:
    datasets[c], total = fetch_doodles(c, n)
    print(f"{c}: kept {datasets[c].shape}, full file has {total} doodles")
    # "kept (2000, 784)" means 2000 doodles with 784 pixels each

# --- Step 2: show the first 6 doodles of each animal ---
# Make a grid of small plots: one row per animal, 6 columns
fig, axes = plt.subplots(len(categories), 6, figsize=(9, 6))
for row, c in enumerate(categories):
    for col in range(6):
        # Each doodle is stored as a flat row of 784 numbers.
        # reshape(28, 28) turns it back into a square picture.
        axes[row, col].imshow(datasets[c][col].reshape(28, 28), cmap="gray_r")
        axes[row, col].axis("off")              # hide the axis numbers
    axes[row, 0].set_title(c, loc="left")       # label each row with the animal's name

plt.tight_layout()                              # stop the plots from overlapping
plt.savefig("sample_doodles.png", dpi=150)      # save the picture as a file for your report
plt.show()                                      # also display it on screen

# --- Step 3: split the data into training and test sets ---
# Combine all animals into one big array.
# Pixel values are scaled from 0-255 down to 0-1, and each doodle gets a label number (0 = cat, 1 = dog, ...)
X = np.concatenate([datasets[c] for c in categories]).astype("float32") / 255.0
y = np.concatenate([np.full(len(datasets[c]), i) for i, c in enumerate(categories)])

# Keep 80% for training and set aside 20% for testing.
# stratify=y keeps the animals evenly balanced in both groups; random_state makes the split repeatable.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print("train:", X_train.shape, " test:", X_test.shape)

# --- Step 4: template-matching baseline ---
# Build one "template" per animal by averaging all of its training doodles
templates = np.stack([X_train[y_train == i].mean(axis=0) for i in range(len(categories))])

# For each test doodle, measure how far it is from each template (smaller = more similar),
# then guess the animal whose template is closest
dists = ((X_test[:, None, :] - templates[None, :, :]) ** 2).sum(axis=2)
predictions = dists.argmin(axis=1)

# Score the guesses against the true answers
accuracy = (predictions == y_test).mean()                                             # exactly right
top3 = (np.argsort(dists, axis=1)[:, :3] == y_test[:, None]).any(axis=1).mean()       # right answer in best 3
print(f"Template matching accuracy: {accuracy:.1%}")
print(f"Template matching top-3 accuracy: {top3:.1%}")

# --- Step 5: draw the templates and a confusion matrix, then save the picture ---
cm = confusion_matrix(y_test, predictions, normalize="true")   # each row: where that animal's doodles went
fig, axes = plt.subplots(1, len(categories) + 1, figsize=(15, 3))
for i, c in enumerate(categories):
    axes[i].imshow(templates[i].reshape(28, 28), cmap="gray_r")
    axes[i].set_title(f"{c} template")
    axes[i].axis("off")
axes[-1].imshow(cm, cmap="Blues", vmin=0, vmax=1)
axes[-1].set_xticks(range(len(categories)), categories, rotation=45)
axes[-1].set_yticks(range(len(categories)), categories)
axes[-1].set_xlabel("Predicted")
axes[-1].set_ylabel("True")
axes[-1].set_title("Confusion matrix")
plt.tight_layout()
plt.savefig("baseline_results.png", dpi=150)
plt.show()