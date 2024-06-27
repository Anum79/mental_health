#%%writefile voice_app.py

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import speech_recognition as sr
from pydub import AudioSegment
import os
from gtts import gTTS

# Initialize the model and tokenizer
model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=False, revision="main")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Function to convert audio to text
def audio_to_text(audio_filename):
    audio = AudioSegment.from_mp3(audio_filename)
    wav_filename = "temp_audio.wav"
    audio.export(wav_filename, format="wav")

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_filename) as source:
        audio_data = recognizer.record(source)

    try:
        transcript = recognizer.recognize_google(audio_data)
    except sr.RequestError:
        transcript = "Error: API unavailable or unresponsive."
    except sr.UnknownValueError:
        transcript = "Error: Unable to recognize speech."

    os.remove(wav_filename)
    return transcript

# Function to convert text to speech
def text_to_speech(text, filename="output.mp3"):
    tts = gTTS(text, lang='en')
    tts.save(filename)

# Instructions for the assistant
instructions_string = f"""a virtual psychiatrist assistant, communicates in clear, accessible language, escalating to technical depth upon request. \
It reacts to feedback aptly and ends responses with its signature. \
providing concise acknowledgments to brief expressions of gratitude or feedback, \
thus keeping the interaction natural and engaging.
Please respond to the following comment.
"""

# Prompt template
prompt_template = lambda comment: f'''[INST] {instructions_string} \n{comment} \n[/INST]'''

# Streamlit application
st.title("Virtual Psychiatrist Assistant")
st.header("Upload your audio file")

uploaded_file = st.file_uploader("Choose an audio file", type=["mp3"])

if uploaded_file is not None:
    with open("uploaded_audio.mp3", "wb") as f:
        f.write(uploaded_file.getbuffer())

    transcript = audio_to_text("uploaded_audio.mp3")
    st.write("Transcript:", transcript)

    prompt = prompt_template(transcript)
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
    response = tokenizer.decode(output[0])

    start_token = '[/INST]'
    end_token = '</s>'
    start_index = response.find(start_token) + len(start_token)
    end_index = response.find(end_token, start_index)
    answer = response[start_index:end_index].strip()

    st.write("Assistant's Response:", answer)

    text_to_speech(answer, filename="response.mp3")

    audio_file = open("response.mp3", "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/mp3")
    st.download_button(label="Download Response Audio", data=audio_bytes, file_name="response.mp3", mime="audio/mp3")
