from flask import Flask, request, jsonify, render_template,redirect,url_for,session,flash
from speech_funct import (
    transcribe_audio,
    convert_to_pcm_wav,
    translate_m2m,
    interactive_transcribe_translate_tts
)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('sts.html')

@app.route("/process", methods=["POST"])
def process():
    if "audio" not in request.files:
        return jsonify({"error": "No audio uploaded"}), 400

    audio_file = request.files["audio"]
    src_choice = request.form.get("src_lang", "English")  # default
    tgt_choice = request.form.get("tgt_lang", "Hausa")    # default

  
    temp_path = "temp_audio.webm"
    audio_file.save(temp_path)                          

    
    wav_path = convert_to_pcm_wav(temp_path)


    result = interactive_transcribe_translate_tts(wav_path, src_choice, tgt_choice)

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0",port="8000",debug=True)