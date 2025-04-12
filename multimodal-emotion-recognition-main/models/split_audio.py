from pydub import AudioSegment
import os

# Define the directory containing your audio files
audio_folder = "D:/DL/drive_data/actor10"
output_folder = "D:/DL/drive_data/actor10"

# The duration of each segment in milliseconds
segment_duration_ms = 3000

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate over each file in the directory
for filename in os.listdir(audio_folder):
    if filename.endswith(".wav"):  # or any other audio format
        file_path = os.path.join(audio_folder, filename)
        audio = AudioSegment.from_file(file_path)

        # Calculate the number of segments
        n_segments = len(audio) // segment_duration_ms

        # Split the audio and export segments
        for i in range(n_segments):
            start_time = i * segment_duration_ms
            end_time = start_time + segment_duration_ms
            segment = audio[start_time:end_time]
            segment.export(os.path.join(output_folder, f"04_01_{i}.wav"), format="wav")
