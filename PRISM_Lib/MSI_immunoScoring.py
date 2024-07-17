# Necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyimzml.ImzMLParser import ImzMLParser
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import zoom
import warnings
import joblib

# Function to generate a TIC map from an imzML file
def generate_tic_map(imzml_file, mzs_range=(600, 1000), sigma=0.5, new_resolution=4.0):
    # Read the imzML file
    p = ImzMLParser(imzml_file)
    spectra = []

    for idx, (x, y, z) in enumerate(p.coordinates):
        mzs, intensities = p.getspectrum(idx)
        mask = (mzs >= mzs_range[0]) & (mzs <= mzs_range[1])
        mzs = mzs[mask]
        intensities = intensities[mask]
        spectra.append([mzs, intensities, (x, y, z)])

    x_coords = [x for _, _, (x, _, _) in spectra]
    y_coords = [y for _, _, (_, y, _) in spectra]
    max_x, max_y = max(x_coords), max(y_coords)  # Define these based on your data

    # Create an empty array to store the sum of intensities for each pixel
    sum_intensities = np.zeros((max_y + 1, max_x + 1))

    # Accumulate the sum of intensities for each pixel
    for mzs, intensities, (x, y, z) in spectra:
        sum_intensities[y, x] += np.sum(intensities)

    # Get the indices of non-zero pixels
    real_pixels = np.where(sum_intensities != 0)

    # Crop the sum_intensities array to contain only non-zero pixels
    sum_intensities = sum_intensities[min(real_pixels[0]): max(real_pixels[0]) + 1,
                                      min(real_pixels[1]): max(real_pixels[1]) + 1]

    # Gaussian smoothing if needed
    smoothed_image = gaussian_filter(sum_intensities, sigma=sigma)

    # Increase spatial resolution using interpolation if needed
    zoomed_image = zoom(smoothed_image, new_resolution, order=3)

    # Plot the TIC map
    plt.imshow(zoomed_image, cmap='jet')
    plt.colorbar()
    plt.title("TIC MAP")
    plt.show()
    
# Function to generate label maps based on an imzML file and a trained model
def generate_label_maps(imzml_file, model_file, mass_range=(600, 1000), max_intensity_size=4000, sigma=0.5, new_resolution=3.0, real_pixel_threshold=10000):
    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Load the trained model
    model = joblib.load(model_file)

    # Read the imzML file and preprocess the data
    p = ImzMLParser(imzml_file)
    my_spectra = []  # Initialize my_spectra list
    class_names = model.classes_
    predicted_scores = []

    # Initialize variables for max_x and max_y
    max_x = 0
    max_y = 0

    for idx, (x, y, z) in enumerate(p.coordinates):
        mzs, intensities = p.getspectrum(idx)
        # Filter values within the specified mass range
        mask = (mzs >= mass_range[0]) & (mzs <= mass_range[1])
        mzs = mzs[mask]
        intensities = intensities[mask]
        if len(mzs) == 0:  # Skip spectra with no data in the specified mass range
            continue
        if len(intensities) > max_intensity_size:
            padded_intensities = intensities[:max_intensity_size]
        else:
            padding = max_intensity_size - len(intensities)
            padded_intensities = np.pad(intensities, (0, padding), mode='constant')
        padded_intensities = padded_intensities.reshape(1, -1)  # Reshape to 2D array
        scores = model.predict_proba(padded_intensities)
        predicted_scores.append(scores[0])  # Store the predicted scores for this spectrum
        my_spectra.append((x, y, padded_intensities))  # Collect the spectra data

        # Update max_x and max_y based on coordinates
        max_x = max(max_x, x)
        max_y = max(max_y, y)

    if max_x == 0 or max_y == 0:
        print("No spectra found within the specified mass range.")
        return

    # Create an empty array to store the sum of intensities for each pixel
    sum_intensities = np.zeros((max_y + 1, max_x + 1))

    # Iterate through the spectra and update sum_intensities
    for spectrum in my_spectra:
        x, y, intensities = spectrum
        sum_intensities[y, x] += np.sum(intensities)

    # Get indices of non-zero pixels based on the real_pixel_threshold
    real_pixels = np.where(sum_intensities > real_pixel_threshold)

    # Define colormap
    cmap = 'jet'

    for label_idx, label in enumerate(class_names):
        label_map = np.zeros((max_y + 1, max_x + 1))
        for pixel_idx in range(np.array(real_pixels).shape[1]):
            y, x = real_pixels[0][pixel_idx], real_pixels[1][pixel_idx]
            label_map[y, x] = predicted_scores[pixel_idx][label_idx]  # Use predicted_scores for label map

        # Apply Gaussian smoothing to the label map if needed
        smoothed_label_map = gaussian_filter(label_map, sigma=sigma)

        # Increase the spatial resolution using interpolation if needed
        zoomed_label_map = zoom(smoothed_label_map, new_resolution, order=3)

        plt.figure()
        plt.imshow(zoomed_label_map, cmap=cmap, vmin=0, vmax=1)
        plt.colorbar()
        plt.title(label)

    plt.show()
    
# Function to calculate label ratios based on an imzML file and a trained model
def calculate_label_ratios(imzml_file, model_file, mass_range=(600, 1000), max_intensity_size=4000, sigma=0, new_resolution=3, real_pixel_threshold=10000):
        # Suppress warnings
    warnings.filterwarnings("ignore")

    # Load the trained model
    model = joblib.load(model_file)

    # Read the imzML file and preprocess the data
    p = ImzMLParser(imzml_file)
    my_spectra = []  # Initialize my_spectra list
    class_names = model.classes_
    predicted_scores = []

    # Initialize variables for max_x and max_y
    max_x = 0
    max_y = 0

    for idx, (x, y, z) in enumerate(p.coordinates):
        mzs, intensities = p.getspectrum(idx)
        # Filter values within the specified mass range
        mask = (mzs >= mass_range[0]) & (mzs <= mass_range[1])
        mzs = mzs[mask]
        intensities = intensities[mask]
        if len(mzs) == 0:  # Skip spectra with no data in the specified mass range
            continue
        if len(intensities) > max_intensity_size:
            padded_intensities = intensities[:max_intensity_size]
        else:
            padding = max_intensity_size - len(intensities)
            padded_intensities = np.pad(intensities, (0, padding), mode='constant')
        padded_intensities = padded_intensities.reshape(1, -1)  # Reshape to 2D array
        scores = model.predict_proba(padded_intensities)
        predicted_scores.append(scores[0])  # Store the predicted scores for this spectrum
        my_spectra.append((x, y, padded_intensities))  # Collect the spectra data

        # Update max_x and max_y based on coordinates
        max_x = max(max_x, x)
        max_y = max(max_y, y)

    if max_x == 0 or max_y == 0:
        print("No spectra found within the specified mass range.")
        return None, None, None  # Return None if no data is found

    # Create an empty array to store the sum of intensities for each pixel
    sum_intensities = np.zeros((max_y + 1, max_x + 1))

    # Iterate through the spectra and update sum_intensities
    for spectrum in my_spectra:
        x, y, intensities = spectrum
        sum_intensities[y, x] += np.sum(intensities)

    # Get indices of non-zero pixels based on the real_pixel_threshold accorting to the TIC map 
    real_pixels = np.where(sum_intensities > real_pixel_threshold)
    
    
    
    # Initialize dictionaries to store the sum of scores for each label
    label_sums = {}
    for label in class_names:
        label_sums[label] = 0.0

    # Calculate the sum of scores for each label
    for pixel_idx in range(len(predicted_scores)):
        for label_idx, label in enumerate(class_names):
            label_sums[label] += predicted_scores[pixel_idx][label_idx]

    # Calculate the ratios of each label
    ratios = {}
    for label in class_names:
        ratios[label] = label_sums[label] / np.sum(list(label_sums.values()))

    # Convert ratios dictionary to DataFrame
    df_ratios = pd.DataFrame.from_dict(ratios, orient='index', columns=['Ratio'])

    return df_ratios