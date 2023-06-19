import os  # Manipulate operating system interfaces.
import random  # Random variable generators.
import streamlit as st  # Streamlit.
import sys  # System-specific parameters and functions.
import librosa  # Music and audio analysis.
import numpy as np  # Data wrangling.
import os  # Manipulate operating system interfaces.
import pandas as pd  # Data handling.
import soundfile as sf
import python_speech_features 
from python_speech_features import logfbank
from python_speech_features import mfcc
from keras.models import Sequential, model_from_json
# Python-dotenv reads key-value pairs from a .env file and can set them as environment variables.
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

# Python-dotenv reads key-value pairs from a .env file and can set them as environment variables.
from dotenv import load_dotenv

# This Python module provides bindings for the PortAudio library and a few convenience functions to play and record NumPy arrays containing audio signals.
from sounddevice import rec, wait

# Write a NumPy array as a .wav file.
from scipy.io.wavfile import write

load_dotenv()


# Show the model architecture

# Keras optimiser

# List of emotions the model was trained on.

classes = ['Angry', 'Happy', 'Neutral', 'Sad','Surprise']  # Replace with your own class labels
    

def noise(data):
    data = np.array(data, dtype=float)  # Convert data to a numeric array
    noise_amp = 0.04 * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.70):
    return librosa.effects.time_stretch(data,rate=rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

def higher_speed(data, speed_factor = 1.25):
    return librosa.effects.time_stretch(data,rate=speed_factor)

def lower_speed(data, speed_factor = 0.75):
    return librosa.effects.time_stretch(data,rate=speed_factor)




def predict_audio_class(audio_path):
    # Preprocess the audio

    # Make predictions

    predicted_class = classes[np.argmax(audio_path)]

    return predicted_class


def extract_features(data):
    
        
    result = np.array([])
    try:
        mfccs = librosa.feature.mfcc(y=data, sr=22050, n_mfcc=58)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        result = np.array(mfccs_processed)
        #st.write("Ok")
    except Exception as e:
        st.write("Error: ", str(e))
        #st.write("Ok not")
    return result
    

    #result = np.array([])

    #mfccs = python_speech_features.mfcc(signal=data, samplerate=22050, numcep=58)
    #mfccs_processed = np.mean(mfccs.T,axis=1)
    #result = np.array(mfccs_processed)
    #return result
    

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    #st.write("Hello1")

    data, sample_rate = sf.read(path)

    #st.write("Hello2")
    #without augmentation
    res1 = extract_features(data)
    result = np.array(res1)
    #st.write(result.shape)

    #st.write("Hello3")
    #noised
    noise_data = noise(data)
    #st.write("Hello4")
    res2 = extract_features(noise_data)
    #st.write("Hello5")
    result = np.vstack((result, res2)) # stacking vertically

    #st.write("Hello6")
    #stretched
    #stretch_data = stretch(data)
    #res3 = extract_features(stretch_data)
    #result = np.vstack((result, res3))
    #st.write("Hello7")
    #shifted
    shift_data = shift(data)
    res4 = extract_features(shift_data)
    result = np.vstack((result, res4))
    #st.write("Hello8")
    #pitched
    #pitch_data = pitch(data, sample_rate)
   # res5 = extract_features(pitch_data)
   # result = np.vstack((result, res5))
   # st.write("Hello9")
    #speed up
   # higher_speed_data = higher_speed(data)
   # res6 = extract_features(higher_speed_data)
   # result = np.vstack((result, res6))
   # st.write("Hello10")
    #speed down
   # lower_speed_data = higher_speed(data)
   # res7 = extract_features(lower_speed_data)
   # result = np.vstack((result, res7))
   # st.write("Hello11")
    return result


# Absolute paths must be used.


# Use local CSS.
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Load CSS.
local_css("styles/style.css")

# Prompts used in training data.
prompts = [
    "Kids are talking by the door",
    "Dogs are sitting by the door",
    "It's eleven o'clock",
    "That is exactly what happened",
    "I'm on my way to the meeting",
    "I wonder what this is about",
    "The airplane is almost full",
    "Maybe tomorrow it will be cold",
    "I think I have a doctor's appointment",
    "Say the word apple",
]

emotion_dict = {
    "angry": "angry 😡",
    "happy": "happy 😆",
    "neutral": "neutral 🙂",
    "sad": "sad 😢",
    "surprise": "surprised 😳",
}

# Session states.
if "initial_styling" not in st.session_state:
    st.session_state["initial_styling"] = True

if "particle" not in st.session_state:
    st.session_state["particle"] = "👋🏻"

if "prompt" not in st.session_state:
    st.session_state["prompt"] = ""

if "emotion" not in st.session_state:
    st.session_state["emotion"] = ""

if "is_prompt" not in st.session_state:
    st.session_state["is_prompt"] = False

if "is_emotion" not in st.session_state:
    st.session_state["is_emotion"] = False

if "is_first_time_prompt" not in st.session_state:
    st.session_state["is_first_time_prompt"] = True

# Emotion emoji animation.
def styling(particle):
    return st.markdown(
        f"""
      <div class="snowflake">{particle}</div>
      <div class="snowflake">{particle}</div>
      <div class="snowflake">{particle}</div>
      <div class="snowflake">{particle}</div>
      <div class="snowflake">{particle}</div>

      <div class='box'>
        <div class='wave -one'></div>
        <div class='wave -two'></div>
        <div class='wave -three'></div>
      </div>
    """,
        unsafe_allow_html=True,
    )


# Bootstrap cards with reference to CSS.
st.markdown(
    """<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/css/bootstrap.min.css"
        integrity="sha384-Zenh87qX5JnK2Jl0vWa8Ck2rdkQ2Bzep5IDxbcnCeuOxjzrPF/et3URy9Bv1WTRi" crossorigin="anonymous">
    """,
    unsafe_allow_html=True,
)


def make_grid(rows, cols):
    grid = [0] * rows
    for i in range(rows):
        with st.container():
            grid[i] = st.columns(cols)
    return grid


# Title.
title = f"""<p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.3rem;">
            Speech Emotion Analyzer</p>"""
st.markdown(title, unsafe_allow_html=True)

# Image.
image = "https://t4.ftcdn.net/jpg/03/27/36/95/360_F_327369570_CAxxxHHLvjk6IJ3wGi1kuW6WTtqjaMpc.jpg"
st.image(image, use_column_width=True)

# Header.
header = f"""<p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 1.7rem;">
            HiveMind</p>"""
st.markdown(header, unsafe_allow_html=True)



# Make the prediction.
def predict(path):
    #st.write("K")

    model = load_model('/Users/sanz/vera/frontend/Emotion_Model_conv1d.h5')
    
    #st.write("Loaded model from disk")
 

    classes = ['Angry', 'Happy', 'Neutral', 'Sad','Surprise']  # Replace with your own class labels
    #st.write("Loaded model from disk2")
 
    pred=(model.predict(path))
    
    #st.write("Loaded model from disk3")
    #st.write(pred)
 

    predicted_class = classes[np.argmax(pred)]
    return predicted_class


# Prompt button.
def prompt_btn():
    if not (st.session_state["is_first_time_prompt"]):
        styling(particle=st.session_state["particle"])

    prompt = '"' + random.choice(prompts) + '"'
    st.session_state["prompt"] = prompt

    st.markdown(
        f"""
            <p align="center" style="font-family: monospace; color: #ffffff; font-size: 2rem;">
            {st.session_state["prompt"]}</p>
        """,
        unsafe_allow_html=True,
    )

    if not (st.session_state["is_first_time_prompt"]):
        st.markdown(
            f"""
                <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">
                Try to sound {emotion_dict.get(st.session_state["emotion"])}</p>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
                <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">
                HiveMind</p>
            """,
            unsafe_allow_html=True,
        )


# Emotion button.
def emotion_btn():
    st.session_state["initial_styling"] = False
    st.session_state["is_first_time_prompt"] = False

    emotion = random.choice(list(emotion_dict))
    partition = emotion_dict.get(emotion).split(" ")
    emotion = partition[0]
    st.session_state["emotion"] = emotion

    if st.session_state["emotion"] == "disgusted":
        st.session_state["emotion"] = "disgust"

    if st.session_state["emotion"] == "scared":
        st.session_state["emotion"] = "fear"

    if st.session_state["emotion"] == "surprised":
        st.session_state["emotion"] = "surprise"

    particle = partition[1]
    st.session_state["particle"] = particle
    styling(particle=st.session_state["particle"])

    st.markdown(
        f"""
            <p align="center" style="font-family: monospace; color: #ffffff; font-size: 2rem;">
            {st.session_state["prompt"]}</p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
            <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">
            Try to sound {emotion_dict.get(st.session_state["emotion"])}</p>
        """,
        unsafe_allow_html=True,
    )


# Record button.
def record_btn():
    fs = 44100  # Sample rate.
    seconds = 3  # Duration of recording.

    with st.spinner(f"Recording for {seconds} seconds ...."):
        myrecording = rec(int(seconds * fs), samplerate=fs, channels=1)
        wait()  # Wait until recording is finished.

        write(
              "recording.wav", fs, myrecording
        )  # Save as .wav file.
        st.success("Recording completed.")


# Play button.
def play_btn():  # Play the recorded audio.
    styling(particle=st.session_state["particle"])
    st.markdown(
        f"""
            <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2rem;">
            {st.session_state["prompt"]}</p>
        """,
        unsafe_allow_html=True,
    )
    if not (st.session_state["is_first_time_prompt"]):
        st.markdown(
            f"""
                <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">
                Try to sound {emotion_dict.get(st.session_state["emotion"])}</p>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
                <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">
                HiveMind</p>
                <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">Your recorded audio.</p>
            """,
            unsafe_allow_html=True,
        )
    try:  # Load audio file.
        audio_file = open("recording.wav", "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes)

    except:
        st.write("Please record sound first.")


# Classify button.
def classify_btn():
    try:
        wav_path = "recording.wav"
        #st.write("Opening")

        audio_features = get_features(wav_path)
        #st.write(audio_features)
        x_test2 = np.expand_dims(audio_features, axis=2)                 
        #st.write(x_test2.shape)
        #st.write("Features getting")
        emotion = predict(x_test2)
        #st.write("Emo getting")
        st.write(emotion)

        if emotion == "happy":
            emotion = "Happy"

        if emotion == "angry":
            emotion = "Anger"

        if emotion == "surprise":
            emotion = "surprised"

        if st.session_state["emotion"] == "disgust":
            st.session_state["emotion"] = "disgusted"

        if st.session_state["emotion"] == "fear":
            st.session_state["emotion"] = "scared"

        if st.session_state["emotion"] == "surprise":
            st.session_state["emotion"] = "surprised"

        if st.session_state["emotion"] != "":
            if emotion in st.session_state["emotion"]:
                st.session_state["particle"] = "😆"
                styling(particle=st.session_state["particle"])
                st.markdown(
                    f"""
                            <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">
                            You tried to sound {st.session_state["emotion"].upper()} and you sounded {emotion.upper()}</p>
                            <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">Well done!👍</p>
                        """,
                    unsafe_allow_html=True,
                )

                try:  # Load audio file.
                    audio_file = open(
                         "recording.wav", "rb"
                    )
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes)

                except:
                    st.write("Please record sound first.")
                st.balloons()

            else:
                st.session_state["particle"] = "😢"
                styling(particle=st.session_state["particle"])
                st.markdown(
                    f"""
                            <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">
                            You tried to sound {st.session_state["emotion"].upper()} however you sounded {emotion.upper()}👎</p>
                        """,
                    unsafe_allow_html=True,
                )

                try:  # Load audio file.
                    audio_file = open(
                        "recording.wav", "rb"
                    )
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes)

                except:
                    st.write("Please record sound first.")
        else:
            st.markdown(
                f"""
                    <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">Emotion Recognized</p>
                    """,
                unsafe_allow_html=True,
            )
    except:
        st.write("Something went wrong. Please try again.")


# User Interface.
if st.session_state["initial_styling"]:
    styling(particle=st.session_state["particle"])

# Create custom grid.
grid1 = make_grid(3, (12, 12, 4))

# Prompt Button.
#prompt = grid1[0][0].button("Prompt")
#if prompt or st.session_state["is_prompt"]:
#    st.session_state["is_emotion"] = False
#    prompt_btn()

# Emotion Button.
#emotion = grid1[0][2].button("Emotion")
#if emotion or st.session_state["is_emotion"]:
#    st.session_state["is_prompt"] = False
#    emotion_btn()

# Create custom grid.
grid2 = make_grid(3, (12, 12, 4))


# Record Button.
record = grid2[0][0].button("Record")
if record:
    record_btn()

# Play Button.
play = grid2[0][1].button("Play")
if play:
    play_btn()

# Classify Button.
classify = grid2[0][2].button("Classify")
if classify:
    classify_btn()

# GitHub repository of project.
st.markdown(
    f"""
        <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 1rem;"><b> Thanks
        </b>
        </p>
   """,
    unsafe_allow_html=True,
)
