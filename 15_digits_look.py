import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
print(digits.images.shape)
print(digits.target[:10])
print(digits.images[0])        # the raw numbers behind the first image

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap="gray")
    ax.set_title(f"Label: {digits.target[i]}")
    ax.axis("off")
plt.show()