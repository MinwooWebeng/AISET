import numpy as np
import librosa
import soundfile as sf
from pedalboard import Pedalboard, Compressor, Distortion, Delay, Reverb, PeakFilter
import os

# Define available plugins with parameter ranges
available_plugins = {
    'PeakFilter': (     #EQ with one peak
        PeakFilter,
        {
            'cutoff_frequency_hz': (40.0, 22000.0),  # Range: 40Hz to 22kHz
            'gain_db': (-12, 12),         # Range: -12 to 12dB
            'q': (0.1, 100.0)    # Range: 0.1 to 100
        }
    ),
    'Compressor': (     #Compressor
        Compressor,
        {
            'threshold_db': (-40.0, 0.0),  # Range: -40 to 0 dB
            'ratio': (1.0, 20.0),         # Range: 1:1 to 20:1
            'attack_ms': (1.0, 100.0),    # Range: 1 to 100 ms
            'release_ms': (10.0, 1000.0)  # Range: 10 to 1000 ms
        }
    ),
    'Distortion': (
        Distortion,
        {
            'drive_db': (0.0, 40.0)       # Range: 0 to 40 dB
        }
    ),
    'Delay': (
        Delay,
        {
            'delay_seconds': (0.01, 1.0), # Range: 10 ms to 1 s
            'feedback': (0.0, 0.9),        # Range: 0 to 90%
            'mix': (0.0, 1.0)        # Range: 0 to 100%
        }
    ),
    'Reverb': (
        Reverb,
        {
            'room_size': (0.0, 1.0),      # Range: 0 (small) to 1 (large)
            'damping': (0.0, 1.0)         # Range: 0 to 1
        }
    ),
    'Gain': (
        Gain,
        {
            'gain_db': (-6.0, 6.0)        # Range: -6 to 6 dB
        }
    )
}

def FX_to_Audio(parameters, input_audio_path, plugins_to_use = \
                ['PeakFilter', 'Compressor', 'Distortion', 'Delay', 'Reverb', 'Gain',\
                 'PeakFilter', 'Compressor', 'Distortion', 'Delay', 'Reverb', 'Gain',\
                 'PeakFilter', 'Compressor', 'Distortion', 'Delay', 'Reverb', 'Gain']):
    """
    Apply a sequence of audio effects to an input audio file.
    
    Args:
        plugins_to_use (list of str): List of plugin names to apply in order.
        parameters (list of float): List of normalized parameters (0.0 to 1.0) for all plugins in order.
        input_audio_path (str): Path to the input audio file.

    Returns:
        np.ndarray: Processed audio signal.
        int: Sampling rate of the processed audio.
    """
    # Verify plugins and calculate total expected parameters
    total_expected_params = 0
    parameter_ranges = []
    for plugin_name in plugins_to_use:
        if plugin_name not in available_plugins:
            raise ValueError(f"Plugin '{plugin_name}' is not available.")
        plugin_class, param_info = available_plugins[plugin_name]
        total_expected_params += len(param_info)
        parameter_ranges.extend(param_info.values())

    # Check if the number of parameters matches the total expected
    if len(parameters) != total_expected_params:
        raise ValueError(f"Incorrect number of parameters: expected {total_expected_params}, got {len(parameters)}.")

    # Map normalized parameters to their actual ranges
    mapped_parameters = [
        (p_min + value * (p_max - p_min))
        for value, (p_min, p_max) in zip(parameters, parameter_ranges)
    ]

    # Load audio file
    audio, sample_rate = librosa.load(input_audio_path, sr=None, mono=True)

    # Initialize the pedalboard
    board = Pedalboard([])

    # Apply parameters to plugins
    param_index = 0  # Tracks current position in the mapped_parameters list
    for plugin_name in plugins_to_use:
        plugin_class, param_info = available_plugins[plugin_name]
        plugin_instance = plugin_class()

        # Set parameters dynamically
        for param_key in param_info:
            setattr(plugin_instance, param_key, mapped_parameters[param_index])
            param_index += 1

        board.append(plugin_instance)

    # Process the audio
    processed_audio = board(audio, sample_rate)

    return processed_audio, sample_rate

def create_dataset(native_dataset_path, audio_files_path, output_dir):
    """
    Create a dataset by applying plugins and parameters to audio files.

    Args:
        native_dataset_path (str): Path to the 'native_dataset.txt' file.
        audio_files_path (str): Path to the 'audio_files.txt' file.
        output_dir (str): Directory to save the processed audio files.

    Returns:
        None
    """
    # Check if the output directory exists, create it if not
    os.makedirs(output_dir, exist_ok=True)

    # Read data from the text files
    with open(native_dataset_path, 'r') as native_file:
        native_data = native_file.readlines()
    
    with open(audio_files_path, 'r') as audio_file:
        audio_files = audio_file.readlines()

    # Check if the number of rows matches
    if len(native_data) != len(audio_files):
        raise ValueError("Mismatch between the number of rows in 'native_dataset.txt' and 'audio_files.txt'.")

    # Process each row
    for idx, (native_row, audio_row) in enumerate(zip(native_data, audio_files)):
        # Parse plugins and parameters
        native_row = native_row.strip()  # Remove leading/trailing whitespace
        audio_row = audio_row.strip()

        # Example native_row format: "Compressor,Reverb;0.5,0.8,0.2,0.6,0.7,0.3"
        try:
            plugin_part, param_part = native_row.split(';')
            plugins = plugin_part.split(',')
            parameters = list(map(float, param_part.split(',')))
        except ValueError:
            raise ValueError(f"Invalid format in row {idx + 1} of 'native_dataset.txt': {native_row}")

        # Apply the effects using FX_to_Audio
        try:
            processed_audio, sr = FX_to_Audio(parameters, audio_row, plugins)
        except Exception as e:
            print(f"Error processing file {audio_row} with plugins {plugins}: {e}")
            continue

        # Save the processed audio file
        output_path = os.path.join(output_dir, f"processed_audio_{idx + 1}.wav")
        sf.write(output_path, processed_audio, sr)

        print(f"Processed audio saved to {output_path}")

# Example usage
if __name__ == "__main__":
    create_dataset(
        native_dataset_path="./native_dataset.txt",
        audio_files_path="./audio_files.txt",
        output_dir="./datasets/processed"
    )