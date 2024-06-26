import numpy as np
import os


def read_spectral_data(hdr_path, raw_path):
    # Read HDR file for image dimensions information
    with open(hdr_path, 'r', encoding='latin1') as hdr_file:
        lines = hdr_file.readlines()
        height = width = bands = 0
        for line in lines:
            if line.startswith('lines'):
                height = int(line.split()[-1])
            elif line.startswith('samples'):
                width = int(line.split()[-1])
            elif line.startswith('bands'):
                bands = int(line.split()[-1])

    # Read spectral data from RAW file
    raw_image = np.fromfile(raw_path, dtype='uint16')
    # Initialize the image with the actual read dimensions
    formatImage = np.zeros((height, width, bands))

    for row in range(height):
        for dim in range(bands):
            formatImage[row, :, dim] = raw_image[(dim + row * bands) * width:(dim + 1 + row * bands) * width]

    # Ensure the image is 30x30x224 by cropping or padding
    target_height, target_width, target_bands = 30, 30, 224
    # Crop or pad height
    if height > target_height:
        formatImage = formatImage[:target_height, :, :]
    elif height < target_height:
        pad_height = target_height - height
        formatImage = np.pad(formatImage, ((0, pad_height), (0, 0), (0, 0)), mode='constant', constant_values=0)

    # Crop or pad width
    if width > target_width:
        formatImage = formatImage[:, :target_width, :]
    elif width < target_width:
        pad_width = target_width - width
        formatImage = np.pad(formatImage, ((0, 0), (0, pad_width), (0, 0)), mode='constant', constant_values=0)

    # Crop or pad bands if necessary (usually bands should not change)
    if bands > target_bands:
        formatImage = formatImage[:, :, :target_bands]
    elif bands < target_bands:
        pad_bands = target_bands - bands
        formatImage = np.pad(formatImage, ((0, 0), (0, 0), (0, pad_bands)), mode='constant', constant_values=0)

    return formatImage


# Specify the directory containing the HDR and RAW files
directory = r'D:\project\supermachine--tomato-passion_fruit\20240529RGBtest3\xs\光谱数据3030'

# Initialize a list to hold all the spectral data arrays
all_spectral_data = []

# Loop through each data set (assuming there are 40 datasets)
for i in range(1, 101):
    hdr_path = os.path.join(directory, f'{i}.HDR')
    raw_path = os.path.join(directory, f'{i}')

    # Read data
    spectral_data = read_spectral_data(hdr_path, raw_path)
    all_spectral_data.append(spectral_data)

# Stack all data into a single numpy array
all_spectral_data = np.stack(all_spectral_data)
print(all_spectral_data.shape)  # This should print (40, 30, 30, 224)