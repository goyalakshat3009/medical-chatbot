# voiceBot UI with Gradio
import os
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from braindoctor import encode_image, analyze_image_with_query
from voice_of_patient import record_audio, transcribe_with_groq
from voice_of_doctor import text_to_speech_fun, tts_with_elevenlabs


system_prompt = """You have to act as a professional doctor, i know you are not but this is for learning purpose.
What's in this image?. Do you find anything wrong with it medically?
If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in
your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
Donot say 'In the image I see' but say 'With what I see, I think you have ....'
Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot,
Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"""


def process_inputs(audio_filepath, image_filepath):

    speech_to_text_output = transcribe_with_groq(
        GROQ_API_KEY=os.getenv("GROQ_API_KEY"),
        speech_to_text_model="whisper-large-v3",
        audio_file=audio_filepath
    )

    # Handle image input
    if image_filepath:
        encoded_img = encode_image(image_filepath)

        doctor_response = analyze_image_with_query(
            question=system_prompt + speech_to_text_output,
            encoded_img=encoded_img,
            model="meta-llama/llama-4-scout-17b-16e-instruct"
        )
    else:
        doctor_response = "No image provided for me to analyze."

    # Convert doctor response to voice
    tts_with_elevenlabs(
        input_text=doctor_response,
        output_file="final.mp3"
    )

    return speech_to_text_output, doctor_response, "final.mp3"


# Create Gradio interface
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath"),
        gr.Image(type="filepath")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="Doctor's Response"),
        gr.Audio(type="filepath", label="Doctor Voice")
    ],
    title="AI Doctor with Vision and Voice"
)

iface.launch(debug=True, share=True)
#http://127.0.0.1:7860