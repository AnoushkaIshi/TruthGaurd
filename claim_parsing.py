import speech_recognition as sr
from pydub import AudioSegment
from moviepy import AudioFileClip
import io

# Convert audio to WAV format if it's not already in WAV
def convert_to_wav(audio_file):
    audio = AudioSegment.from_file(audio_file)
    wav_file = io.BytesIO()
    audio.export(wav_file, format="wav")
    wav_file.seek(0)
    return wav_file

# Extract audio from video and convert to WAV
def extract_audio_from_video(video_file):
    video_clip = AudioFileClip(video_file)
    audio_path = "extracted_audio.wav"
    video_clip.audio.write_audiofile(audio_path)
    return audio_path

# Transcribe audio using Google Speech Recognition
def speech_to_text(audio_file):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)

    try:
        # Transcribe the audio in English and Hindi
        text = recognizer.recognize_google(audio, language="en-IN")
        return text
    except sr.UnknownValueError:
        return "Could not understand audio."
    except sr.RequestError as e:
        return f"Could not request results; check internet connection. Error: {str(e)}"

