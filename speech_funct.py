import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import tempfile
import whisper
import io
import soundfile as sf
from IPython.display import Audio
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import speech_recognition as sr
from jiwer import wer,cer
import wave
from gtts import gTTS
import warnings
import uuid
import torch
import pygame
from pydub import AudioSegment
import sounddevice as sd
from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    VitsModel
)
warnings.filterwarnings('ignore')

# languages = {
#     "1": ("yo-NG", "Yoruba (Nigeria)", "yor_Latn", "yo", "com.ng"),
#     "2": ("en-US", "English (US)", "eng_Latn", "en", "ca"),
#     "3": ("ha-NG", "Hausa (Nigeria)", "hau_Latn", "ha", "com.ng"),
#     "4": ("ig-NG", "Igbo (Nigeria)", "ibo_Latn", "ig", "com.ng"),
#     "5": ("fr-FR", "French (France)", "fra_Latn", "fr", "fr"),
#     "6": ("de-DE", "German (Germany)", "deu_Latn", "de", "de"),
#     "7": ("zh-CN", "Chinese (Simplified, China)", "zho_Hans", "zh", "com"),
#     "8": ("es-ES", "Spanish (Spain)", "spa_Latn", "es", "es"),
#     "9": ("ar-SA", "Arabic (Saudi Arabia)", "arb_Arab", "ar", "com.sa"),
#     "10": ("pt-PT", "Portuguese (Portugal)", "por_Latn", "pt", "pt"),
#     "11": ("ru-RU", "Russian (Russia)", "rus_Cyrl", "ru", "ru"),
#     "12": ("hi-IN", "Hindi (India)", "hin_Deva", "hi", "co.in"),
#     "13": ("tr-TR", "Turkish (Turkey)", "tur_Latn", "tr", "com.tr"),
# }
lang_profiles = {
    "English": {
        "asr": "en-US",          # SpeechRecognition locale
        "nllb": "eng_Latn",      # NLLB code
        "tts": "en",             # TTS engine code
        "tld": "ca",            # Google TLD
    },
    "Yoruba": {
        "asr": "yo-NG",
        "nllb": "yor_Latn",
        "tts": "yo",
        "tld": "com.ng",
    },
    "Hausa": {
        "asr": "ha-NG",
        "nllb": "hau_Latn",
        "tts": "ha",
        "tld": "com.ng",
    },
    "Igbo": {
        "asr": "ig-NG",
        "nllb": "ibo_Latn",
        "tts": "ig",
        "tld": "com.ng",
    },
    "French": {
        "asr": "fr-FR",
        "nllb": "fra_Latn",
        "tts": "fr",
        "tld": "fr",
    },
    "German": {
        "asr": "de-DE",
        "nllb": "deu_Latn",
        "tts": "de",
        "tld": "de",
    },
    "Chinese": {
        "asr": "zh-CN",
        "nllb": "zho_Hans",
        "tts": "zh",
        "tld": "com",
    },
    "Spanish": {
        "asr": "es-ES",
        "nllb": "spa_Latn",
        "tts": "es",
        "tld": "es",
    },
    "Arabic": {
        "asr": "ar-SA",
        "nllb": "arb_Arab",
        "tts": "ar",
        "tld": "sa",
    },
    "Portuguese": {
        "asr": "pt-PT",
        "nllb": "por_Latn",
        "tts": "pt",
        "tld": "pt",
    },
    "Russian": {
        "asr": "ru-RU",
        "nllb": "rus_Cyrl",
        "tts": "ru",
        "tld": "ru",
    },
    "Hindi": {
        "asr": "hi-IN",
        "nllb": "hin_Deva",
        "tts": "hi",
        "tld": "com",
    },
    "Turkish": {
        "asr": "tr-TR",
        "nllb": "tur_Latn",
        "tts": "tr",
        "tld": "com.tr",
    }
}


recognizer  = sr.Recognizer()

tts_model = VitsModel.from_pretrained("facebook/mms-tts-yor")
tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-yor")
m2m_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
m2m_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

nllb_model_name = "facebook/nllb-200-distilled-600M"
nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_model_name)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name)



def transcribe_audio(file):
    with sr.AudioFile(file) as source:
        audio = recognizer.record(source)
        return recognizer.recognize_google(audio)



def convert_to_pcm_wav(input_path, output_path="converted.wav"):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(output_path, format="wav")
    return output_path


def translate_m2m(src_text: str, src_lang: str = "de", tgt_lang: str = "en") -> str:
    m2m_tokenizer.src_lang = src_lang

    encoded = m2m_tokenizer(src_text, return_tensors="pt")

    generated_tokens = m2m_model.generate(
        **encoded,
        forced_bos_token_id=m2m_tokenizer.get_lang_id(tgt_lang),
        num_beams=5,
        no_repeat_ngram_size=3,
        max_length=100
    )
    return m2m_tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

def speak_yoruba(text: str, samplerate: int = 16000):
    """Generate Yoruba audio file instead of playing on server."""
    inputs = tts_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        wav = tts_model(**inputs).waveform

    audio_data = wav.squeeze().cpu().numpy()

    # Save to static file
    output_path = f"static/output_{uuid.uuid4().hex}.wav"
    sf.write(output_path, audio_data, samplerate)
    return "/" + output_path


def interactive_transcribe_translate_tts(audio_file, src_choice, tgt_choice):
    src_profile = lang_profiles[src_choice]
    tgt_profile = lang_profiles[tgt_choice]

    src_asr  = src_profile["asr"]
    src_nllb = src_profile["nllb"]
    tgt_nllb = tgt_profile["nllb"]
    tgt_tts  = tgt_profile["tts"]
    tgt_tld  = tgt_profile["tld"]

    # --- Step 1: ASR ---
    with sr.AudioFile(audio_file) as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio, language=src_asr)
    except sr.UnknownValueError:
        return {"error": "Could not understand audio"}
    except sr.RequestError as e:
        return {"error": f"ASR request error: {str(e)}"}

    
    nllb_tokenizer.src_lang = src_nllb
    inputs = nllb_tokenizer(text, return_tensors="pt")
    generated_tokens = nllb_model.generate(
        **inputs,
        forced_bos_token_id=nllb_tokenizer.convert_tokens_to_ids(tgt_nllb),
        num_beams=8,
        max_length=200,
        no_repeat_ngram_size=3
    )
    translated_text = nllb_tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    # --- Step 3: TTS ---
    if tgt_nllb == "yor_Latn":
        audio_url = speak_yoruba(translated_text)
    else:
        output_path = f"static/output_{uuid.uuid4().hex}.mp3"
        tts = gTTS(translated_text, lang=tgt_tts, tld=tgt_tld)
        tts.save(output_path)
        audio_url = "/" + output_path

    return {
        "transcribed": text,
        "translated": translated_text,
        "audio_url": audio_url
    }