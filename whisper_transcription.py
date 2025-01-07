import whisper
import tempfile
import os

# Load the Whisper model (This can also be moved to app.py if preferred)
model = whisper.load_model("base")


def transcribe_audio(audio_file):
    # Create a temporary file to save the uploaded audio file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(audio_file.read())
        temp_file_path = temp_file.name

    try:
        # Transcribe the audio using Whisper
        result = model.transcribe(temp_file_path)

        # Return the transcribed text
        return result['text']
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
