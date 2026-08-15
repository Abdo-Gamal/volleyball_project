
import matplotlib.pyplot as plt
import random
import numpy as np


def visualize_samples(dataset, label_map, num_images=5):
    #plt.figure(figsize=(15, 4))
    plt.figure(figsize=(15, 5))   # taller figure

    for i in range(num_images):
        idx = random.randint(0, len(dataset) - 1)
        img, label = dataset[idx]

        # clip (T, 3, H, W) → take middle frame
        if img.dim() == 4:
            img = img[img.shape[0] // 2]

        # unnormalize
        img = img.permute(1, 2, 0).cpu().numpy()
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        img = np.clip(img, 0, 1)

        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        if label_map:
            plt.title(label_map[label])
        
        plt.axis('off')

    plt.show()
