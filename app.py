from flask import Flask, request, jsonify
import whisper
from whisper_transcription import transcribe_audio

app = Flask(__name__)

# Load the Whisper model
model = whisper.load_model("base")


@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio_file = request.files['file']

    # Use the helper function from whisper_transcription.py to handle transcription
    result = transcribe_audio(audio_file)

    return jsonify({"transcription": result})


if __name__ == '__main__':
    app.run(debug=True)
