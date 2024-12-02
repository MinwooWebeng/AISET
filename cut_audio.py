import os
import librosa
import soundfile as sf

def cut_audio(input_folder, output_folder, duration=10):
    """
    Cuts the first `duration` seconds of each audio file in the input folder 
    and saves it to the output folder.

    Args:
        input_folder (str): Path to the folder containing audio files.
        output_folder (str): Path to save the cut audio files.
        duration (int): Duration in seconds to cut from the start of each audio file.

    Returns:
        None
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the input folder
    for file_name in os.listdir(input_folder):
        input_file_path = os.path.join(input_folder, file_name)
        
        # Skip if not a file
        if not os.path.isfile(input_file_path):
            continue

        try:
            # Load the audio file
            audio, sr = librosa.load(input_file_path, sr=None, mono=True)

            # Cut the first `duration` seconds
            cut_audio = audio[:int(duration * sr)]

            # Save the cut audio to the output folder
            output_file_path = os.path.join(output_folder, file_name)
            sf.write(output_file_path, cut_audio, sr)

            print(f"Processed: {file_name} -> {output_file_path}")
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

if __name__ == "__main__":
    # Define input and output folders for clean and processed datasets
    input_clean = './datasets/clean'
    output_clean = './10s_datasets/clean'
    input_processed = './datasets/processed'
    output_processed = './10s_datasets/processed'

    # Cut the first 10 seconds of audio files
    print("Processing clean dataset...")
    cut_audio(input_clean, output_clean, duration=10)

    print("\nProcessing processed dataset...")
    cut_audio(input_processed, output_processed, duration=10)
