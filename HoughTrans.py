from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)
from skimage import data, io, color, feature, filters, exposure
import numpy as np
import matplotlib.pyplot as plt
import os

# Load image
filepath = "easy/flood_easy12.jpg"
name = os.path.splitext(os.path.basename(filepath))[0]

image = io.imread(filepath)

# Pre-processing
img_gray = color.rgb2gray(image) # Convert to grayscale
img_gray = exposure.adjust_gamma(img_gray) # Contrast correction


# Edge detection
edges = feature.canny(img_gray, sigma=1.5) # Canny
# edges = filters.sobel(img_gray) # Use Sobel

# Hough transformation
h, theta, d = hough_line(edges)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10,10))


ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')
ax1.set_axis_off()

ax2.imshow(edges, cmap=plt.cm.gray)
ax2.set_title('Detect edges')
ax2.set_axis_off()

ax3.imshow(h, cmap=plt.cm.gray, aspect=1/1.5)
ax3.set_title('Hough transform')
ax3.set_xlabel('Angles (degrees)')
ax3.set_ylabel('Distance (pixels)')
ax3.axis('image')

ax4.imshow(image, cmap=plt.cm.gray)
rows, cols = edges.shape
for _, angle, dist in zip(*hough_line_peaks(h, theta, d, num_peaks=1)):
    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    y1 = (dist - cols * np.cos(angle)) / np.sin(angle)
    ax4.plot((0, cols), (y0, y1), '-r')
ax4.axis((0, cols, rows, 0))
ax4.set_title('Detected lines')
ax4.set_axis_off()
plt.show()

plt.savefig('easy/' + name + '_result')