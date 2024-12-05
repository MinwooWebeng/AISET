import os
import numpy as np
import librosa
import soundfile as sf
from pedalboard import Convolution
from make_audio import FX_to_Audio  # Import FX_to_Audio from your existing script

def test_genetic_algorithm(input_audio_path, output_directory, parameters, Need_Cabinet=False, cabinet_ir_path=None):
    """
    Apply effects to an input audio file using FX_to_Audio and optionally a cabinet IR.

    Args:
        input_audio_path (str): Path to the input audio file.
        output_directory (str): Directory to save the processed audio file.
        parameters (list of float): List of parameters for FX_to_Audio.
        Need_Cabinet (bool): If True, applies convolution using a cabinet IR.
        cabinet_ir_path (str): Path to the cabinet impulse response file, required if Need_Cabinet is True.

    Returns:
        str: Path to the saved audio file.
    """
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    try:
        # Apply effects using FX_to_Audio
        processed_audio, sample_rate = FX_to_Audio(parameters, input_audio_path)

        # If Need_Cabinet is True, apply convolution with the cabinet IR
        if Need_Cabinet:
            if not cabinet_ir_path or not os.path.exists(cabinet_ir_path):
                raise ValueError("Cabinet IR path is required and must exist if Need_Cabinet is True.")
            
            # Load the cabinet IR file
            cabinet_ir, ir_sample_rate = librosa.load(cabinet_ir_path, sr=sample_rate, mono=True)

            # Ensure IR and audio sample rates match
            if ir_sample_rate != sample_rate:
                raise ValueError("Cabinet IR sample rate does not match the processed audio sample rate.")

            # Apply convolution using Pedalboard's Convolution plugin
            convolver = Convolution(cabinet_ir)
            processed_audio = convolver(processed_audio, sample_rate)

        # Ensure processed_audio is in the correct format
        if processed_audio.ndim == 2:  # Stereo audio
            processed_audio = processed_audio.T  # Transpose to (samples, channels) for saving

        # Generate output file path
        file_name = os.path.basename(input_audio_path)
        output_file = os.path.join(output_directory, file_name)

        # Save the processed audio
        sf.write(output_file, processed_audio, sample_rate)
        print(f"Processed and saved: {output_file}")

        return output_file

    except Exception as e:
        print(f"Error processing {input_audio_path}: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Example parameters for FX_to_Audio (normalized floats)
    test_parameters = [0.5] * 18        # 여기에 GA 결과 입력

    # Test parameters
    input_audio = "./datasets/testdata/guitar_input.wav"  # Path to the input audio file
    output_dir = "./datasets/testdata"  # Output directory for processed files
    cabinet_ir = "./datasets/testdata/412Cab.wav"  # Path to the cabinet IR file

    # Run test with and without the cabinet
    test_genetic_algorithm(input_audio, output_dir, test_parameters, Need_Cabinet=False)
    
    #드라이브(BD2, RAT, TubeScreamer)의 경우 Need_Cabinet=True로 해도 좋음
    #test_genetic_algorithm(input_audio, output_dir, test_parameters, Need_Cabinet=True, cabinet_ir_path=cabinet_ir)