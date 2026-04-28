import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Load the original ground truth data
mat_data = scipy.io.loadmat("Datasets\\PaviaC\\Pavia_gt_original.mat")  # Replace with your actual file path

# Extract the ground truth array
pavia_gt = mat_data['pavia_gt']

# Create a new labeled matrix
new_pavia_gt = np.copy(pavia_gt)
# Merge classes 2 to 9 into class 2
new_pavia_gt[np.isin(pavia_gt, [2, 3, 4, 5, 6, 7, 8, 9])] = 2

# Count pixels in the new class structure
unique_classes, pixel_counts = np.unique(new_pavia_gt, return_counts=True)

# Print new class-wise pixel counts
print("Updated Class-wise Pixel Count:")
for cls, count in zip(unique_classes, pixel_counts):
    print(f'Class {int(cls)}: {count} pixels')

# Save the modified ground truth with preserved metadata
scipy.io.savemat('Datasets\\PaviaC\\Pavia_gt.mat', {
    '__header__': mat_data['__header__'],  # Preserve header
    '__version__': mat_data['__version__'],  # Preserve version
    '__globals__': mat_data['__globals__'],  # Preserve globals
    'pavia_gt': new_pavia_gt  # Save the updated ground truth
})    
