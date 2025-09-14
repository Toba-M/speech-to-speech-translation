# 🌍🎙 Speech2Speech AI – Breaking Language Barriers with AI

**Speech2Speech AI** is a multilingual **speech-to-speech translation system**   
It allows users to **speak in one language** and get **instant spoken translation** in another language all through a simple web app.  

---
This project demonstrates how AI can make real-time, cross-lingual communication possible.  
---

## 🔧 System Pipeline
Speech2Speech AI follows a **three-step pipeline**:

1. **ASR (Automatic Speech Recognition)**  
   - Converts speech to text.  
   - Preprocessing: 16 kHz mono PCM WAV, noise adjustment.  
   - Library: `SpeechRecognition`  

2. **NMT (Neural Machine Translation)**  
   - Translates text between languages.  
   - Models: [Meta M2M100](https://huggingface.co/facebook/m2m100_418M) and [No Language Left Behind (NLLB)](https://ai.facebook.com/research/no-language-left-behind/)  
   - Supports both high- and low-resource languages.  

3. **TTS (Text-to-Speech)**  
   - Converts translated text back into speech.  
   - Hybrid system:  
     - [MMS-TTS (VITS-based)](https://huggingface.co/facebook/mms-tts) → for low-resource languages (e.g., Yoruba).  
     - Google gTTS → for high-resource languages, with accent control via TLDs.  

---

## 🌐 Supported Languages
Currently supports **12+ languages**:  
- English  
- Yoruba  
- Hausa  
- Igbo  
- French  
- German  
- Chinese  
- Spanish  
- Arabic  
- Portuguese  
- Russian  
- Hindi  
- Turkish  

---

## 💻 Web Application
Built with **Flask + JavaScript** to make it interactive:  
- 🎤 Record speech via browser microphone.  
- 🌍 Choose input & output languages with dropdowns.  
- 🔊 Instant playback of translated speech.  

---

## 📸 Demo
👉 [Demo Video Link](#) *(replace with your YouTube/LinkedIn/GitHub video link)*  

---
