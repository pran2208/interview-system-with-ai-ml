import streamlit as st
import PyPDF2
import re
import nltk
import speech_recognition as sr
from elevenlabs import generate, play, set_api_key
import openai
import tempfile

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# 🔐 Your Keys (replace these with your actual keys)
ELEVENLABS_API_KEY = "your-elevenlabs-api-key"
OPENAI_API_KEY = "your-openai-api-key"
set_api_key(ELEVENLABS_API_KEY)
openai.api_key = OPENAI_API_KEY

# ---------------------
# Text Extraction Utils
# ---------------------
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---------------------
# Voice Recognition Utils
# ---------------------
def listen_to_user():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        st.info("🎙️ Listening... please speak now.")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        st.success(f"🗣️ You said: {text}")
        return text
    except sr.UnknownValueError:
        st.error("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError:
        st.error("Speech Recognition service error.")
        return None

# ---------------------
# TTS Output (ElevenLabs)
# ---------------------
def speak_response(text):
    audio = generate(text=text, voice="Rachel", model="eleven_monolingual_v1")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        f.write(audio)
        return f.name

# ---------------------
# ChatGPT Integration
# ---------------------
def get_chatgpt_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or "gpt-4"
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# -------------------
# Streamlit Web App
# -------------------
def main():
    st.set_page_config(page_title="AI ATS + Voice Interview Bot", layout="centered")
    st.title("📄 AI Resume + Voice Interview Assistant")

    resume_text = ""
    uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "txt"])

    if uploaded_file is not None:
        st.success("Resume uploaded successfully!")
        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            resume_text = uploaded_file.read().decode("utf-8")

        resume_text = clean_text(resume_text)
        st.text_area("📄 Extracted Resume", value=resume_text, height=300)

    st.markdown("---")
    st.subheader("🎤 AI Interview Chat")

    if st.button("Start Voice Interview"):
        user_question = listen_to_user()
        if user_question:
            full_prompt = f"I am an interviewer. Here is the candidate's resume:\n{resume_text}\n\nNow answer this question: {user_question}"
            bot_reply = get_chatgpt_response(full_prompt)
            st.markdown(f"**🤖 Bot:** {bot_reply}")
            audio_path = speak_response(bot_reply)
            st.audio(audio_path, format="audio/mp3", autoplay=True)

# --------------
# Run the App
# --------------
if __name__ == "__main__":
    main()


