import os

def write_audio_files_to_txt(input_folder, output_file):
    """
    Reads the names of all files in the specified folder and writes them to a text file.

    Args:
        input_folder (str): Path to the folder containing audio files.
        output_file (str): Path to the output text file.

    Returns:
        None
    """
    # Ensure the folder exists
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"The folder '{input_folder}' does not exist.")
    
    # Get the list of files in the folder
    audio_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    
    # Write the file names to the output text file
    with open(output_file, 'w') as f:
        for audio_file in audio_files:
            # Write the full path to the audio file
            file_path = os.path.join(input_folder, audio_file)
            f.write(f"{file_path}\n")
    
    print(f"File paths written to '{output_file}'.")

# Example usage
if __name__ == "__main__":
    write_audio_files_to_txt(input_folder='./datasets/clean', output_file='audio_files.txt')
