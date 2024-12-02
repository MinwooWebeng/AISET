import numpy as np
import librosa
import soundfile as sf
from pedalboard import Pedalboard, Compressor, Distortion, Chorus, Delay, Reverb

# Available plugins dictionary with parameter info
available_plugins = {
    'Compressor': (Compressor, ['threshold_db', 'ratio', 'attack_ms', 'release_ms']),
    'Distortion': (Distortion, ['drive_db']),
    'Chorus': (Chorus, ['rate_hz', 'depth']),
    'Delay': (Delay, ['delay_seconds', 'feedback']),
    'Reverb': (Reverb, ['room_size', 'damping'])
}

def FX_to_Audio(plugins_to_use, parameters, input_audio_path):
    """
    Apply a sequence of audio effects to an input audio file.

    Args:
        plugins_to_use (list of str): List of plugin names to apply in order.
        parameters (list of float): List of parameters for all plugins in order.
        input_audio_path (str): Path to the input audio file.

    Returns:
        np.ndarray: Processed audio signal.
        int: Sampling rate of the processed audio.
    """
    # Verify plugins and calculate total expected parameters
    total_expected_params = 0
    for plugin_name in plugins_to_use:
        if plugin_name not in available_plugins:
            raise ValueError(f"Plugin '{plugin_name}' is not available.")
        total_expected_params += len(available_plugins[plugin_name][1])

    # Check if the number of parameters matches the total expected
    if len(parameters) != total_expected_params:
        raise ValueError(f"Incorrect number of parameters: expected {total_expected_params}, got {len(parameters)}.")

    # Load audio file
    audio, sample_rate = librosa.load(input_audio_path, sr=None, mono=True)

    # Initialize the pedalboard
    board = Pedalboard([])

    # Apply parameters to plugins
    param_index = 0  # Tracks current position in the parameters list
    for plugin_name in plugins_to_use:
        plugin_class, param_keys = available_plugins[plugin_name]
        plugin_instance = plugin_class()

        # Set parameters dynamically
        for param_key in param_keys:
            setattr(plugin_instance, param_key, parameters[param_index])
            param_index += 1

        board.append(plugin_instance)

    # Process the audio
    processed_audio = board(audio, sample_rate)

    return processed_audio, sample_rate

# Example usage
if __name__ == "__main__":
    # Define plugins to use and their parameters
    plugins = ['Compressor', 'Reverb']
    params = [
        -20.0, 4.0, 10.0, 100.0,  # Compressor parameters: threshold_db, ratio, attack_ms, release_ms
        0.5, 0.8                   # Reverb parameters: room_size, damping
    ]
    input_path = "input_audio.wav"

    # Apply effects
    processed_audio, sr = FX_to_Audio(plugins, params, input_path)

    # Save the processed audio
    sf.write("output_audio.wav", processed_audio, sr)
