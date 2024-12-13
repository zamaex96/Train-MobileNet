import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pycwt as wavelet

# Base paths
base_csv_path = r"C:\CSVs\Seperate"
base_output_path = r"C:\Spectrogram\5pix"

# List of classes from C1 to C9
classes = [f"C{i}" for i in range(1, 10)]


def process_cwt_spectrogram(data, output_folder, sampling_rate=10048):
    """
    Process CSV data and generate Continuous Wavelet Transform spectrograms

    Parameters:
    - data: pandas DataFrame containing the signal data
    - output_folder: path to save generated spectrograms
    - sampling_rate: sampling frequency of the data
    """
    # Exclude the first column
    data = data.iloc[:, 1:]

    # Loop through each column and save its scalogram as a plot
    for column in data.columns:
        y = data[column].values  # Get the signal values of the column
        x = np.arange(len(y)) / sampling_rate  # Generate the time axis based on the sampling rate

        # Perform Continuous Wavelet Transform (CWT)
        mother = wavelet.Morlet(6)
        wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(y, 1 / sampling_rate, wavelet=mother)

        # Create a plot for the current column
        plt.figure(figsize=(10, 6))
        plt.imshow(np.abs(wave), extent=[x[0], x[-1], freqs[-1], freqs[0]], cmap='jet', aspect='auto')

        # Remove the entire axis
        plt.axis('off')

        # Save the plot with tight layout and without white borders
        plot_filename = os.path.join(output_folder, f'{column}.png')
        plt.savefig(plot_filename,
                    format='png',
                    dpi=5,
                    bbox_inches='tight',  # Trim the whitespace around the figure
                    pad_inches=0)  # Remove padding completely

        # Close the plot to free up memory
        plt.close()
        print(f"Saved CWT image for {column} as {plot_filename}")

    print(f"Scalograms saved to folder: {output_folder}")


# Main processing loop
for class_name in classes:
    # Construct full paths
    csv_file_path = os.path.join(base_csv_path, f"{class_name}.csv")
    output_folder = os.path.join(base_output_path, class_name)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    try:
        # Load the CSV file
        data = pd.read_csv(csv_file_path)

        print(f"Processing {class_name}:")
        print(f"Input CSV: {csv_file_path}")
        print(f"Output Folder: {output_folder}")

        # Process the data and generate spectrograms
        process_cwt_spectrogram(data, output_folder)

        print(f"Completed processing for {class_name}\n")

    except FileNotFoundError:
        print(f"Warning: CSV file not found for {class_name}")
    except Exception as e:
        print(f"Error processing {class_name}: {str(e)}")

print("Batch processing complete.")
