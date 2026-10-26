# export_bin.py
# Convierte data/mnist_test.npz en los dos archivos binarios que lee main.cpp:
# mnist_test_images.bin (10000 x 28 x 28 bytes) y mnist_test_labels.bin (10000 bytes).
import argparse
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Exporta el test set de MNIST a .bin para main.cpp")
    parser.add_argument("--npz", default=os.path.join("data", "mnist_test.npz"))
    parser.add_argument("--out", default="build", help="Carpeta donde se corre el ejecutable")
    args = parser.parse_args()

    data = np.load(args.npz)
    images = data["images"].astype(np.uint8)
    labels = data["labels"].astype(np.uint8)

    os.makedirs(args.out, exist_ok=True)
    images.tofile(os.path.join(args.out, "mnist_test_images.bin"))
    labels.tofile(os.path.join(args.out, "mnist_test_labels.bin"))
    print(f"{len(labels)} imágenes {images.shape[1:]} exportadas a {args.out}/")


if __name__ == "__main__":
    main()
