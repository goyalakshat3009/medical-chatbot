import os
from gtts import gTTS
from playsound import playsound
from elevenlabs.client import ElevenLabs
from elevenlabs import save

NEW_ELEVENLABS_API_KEY = os.getenv("NEW_ELEVENLABS_API_KEY")

# gTTS voice
def text_to_speech_fun(input_text, output_file_path):
    tts = gTTS(text=input_text, lang="en", slow=False)
    tts.save(output_file_path)
    playsound(output_file_path)


# ElevenLabs voice
def tts_with_elevenlabs(input_text, output_file):

    client = ElevenLabs(api_key=NEW_ELEVENLABS_API_KEY)

    audio = client.text_to_speech.convert(
        text=input_text,
        voice_id="pNInz6obpgDQGcFmaJgB",
        model_id="eleven_turbo_v2",
        output_format="mp3_22050_32"
    )

    save(audio, output_file)

    return output_file