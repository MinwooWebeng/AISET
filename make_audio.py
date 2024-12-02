import numpy as np
import librosa
import soundfile as sf
from pedalboard import Pedalboard, Compressor, Distortion, Chorus, Delay, Reverb

# Available plugins dictionary to map strings to classes
available_plugins = {
    'Compressor': Compressor,
    'Distortion': Distortion,
    'Chorus': Chorus,
    'Delay': Delay,
    'Reverb': Reverb
}

def FX_to_Audio(plugins_to_use, parameters, input_audio_path):
    """
    Apply a sequence of audio effects to an input audio file.

    Args:
        plugins_to_use (list of str): List of plugin names to apply in order.
        parameters (list of dict): List of dictionaries with parameters for each plugin.
        input_audio_path (str): Path to the input audio file.

    Returns:
        np.ndarray: Processed audio signal.
        int: Sampling rate of the processed audio.
    """
    if len(plugins_to_use) != len(parameters):
        raise ValueError("Each plugin must have a corresponding set of parameters.")

    # Load audio file
    audio, sample_rate = librosa.load(input_audio_path, sr=None, mono=True)

    # Initialize the pedalboard
    board = Pedalboard([])

    # Add plugins to the pedalboard with their parameters
    for plugin_name, param in zip(plugins_to_use, parameters):
        if plugin_name not in available_plugins:
            raise ValueError(f"Plugin '{plugin_name}' is not available.")

        # Create plugin instance and set parameters dynamically
        plugin_class = available_plugins[plugin_name]
        plugin_instance = plugin_class()

        for attr, value in param.items():
            if hasattr(plugin_instance, attr):
                setattr(plugin_instance, attr, value)
            else:
                raise ValueError(f"'{attr}' is not a valid parameter for {plugin_name}.")

        board.append(plugin_instance)

    # Process the audio
    processed_audio = board(audio, sample_rate)

    return processed_audio, sample_rate

# Example usage
if __name__ == "__main__":
    # Define plugins to use and their parameters
    plugins = ['Compressor', 'Reverb']
    params = [
        {'threshold_db': -20.0, 'ratio': 4.0},  # Parameters for Compressor
        {'room_size': 0.5}                     # Parameters for Reverb
    ]
    input_path = "input_audio.wav"

    # Apply effects
    processed_audio, sr = FX_to_Audio(plugins, params, input_path)

    # Save the processed audio
    sf.write("output_audio.wav", processed_audio, sr)
