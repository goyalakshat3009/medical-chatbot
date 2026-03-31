import os
import speech_recognition as sr
from groq import Groq
from dotenv import load_dotenv


# Load API key from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Speech-to-text model
speech_to_text_model = "whisper-large-v3"

def record_audio(file_path, timeout=20, phrase_time_limit=None):  

    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Adjusting for background noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)

        print("Speak now...")
        audio = recognizer.listen(
            source,
            timeout=timeout,
            phrase_time_limit=phrase_time_limit
        )

        print("Recording finished.")

    # Save audio file
    with open(file_path, "wb") as f:
        f.write(audio.get_wav_data())

    print("Audio saved:", file_path)

    return file_path

def transcribe_with_groq(GROQ_API_KEY, speech_to_text_model, audio_file):

    client = Groq(api_key=GROQ_API_KEY)

    with open(audio_file, "rb") as file:

        transcription = client.audio.transcriptions.create(
            file=file,
            model=speech_to_text_model,
            response_format="text"
        )

    return transcription


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":

    audio_file = record_audio(
        file_path="patient_voice.wav",
        timeout=20,
        phrase_time_limit=None
    )

    text = transcribe_with_groq(
        GROQ_API_KEY,
        speech_to_text_model,
        audio_file
    )

    print("\n🧠 Patient said:")
    print(text)