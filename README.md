AI-Based Mock Interview System

An intelligent, adaptive mock interview platform powered by AI. This system simulates real interview environments with dynamic questioning, real-time evaluation, feedback generation, and performance analytics. It is designed for students, job-seekers, and professionals to practice interviews across technical, behavioral, and HR domains.

🚀 Features
✅ AI-Generated Interview Questions

Automatically generates domain-specific interview questions

Supports HR, Behavioral, Technical (CS, IT, ML, Web Dev, etc.)

🎤 Voice or Text Based Interaction

User can answer using text or microphone

AI interviewer responds dynamically

📝 Real-Time Feedback

Evaluates answers based on:

Clarity

Technical accuracy

Soft skills

Confidence & structure

📊 Performance Analytics Dashboard

Score breakdown

Strong vs. weak topics

Suggested improvements

Answer quality graph

🎯 Custom Interview Modes

Beginner

Intermediate

Expert

Company-specific rounds (Google, Amazon, Infosys, etc.)

🧠 AI Model Options

OpenAI GPT

Local LLMs

HuggingFace models

Custom fine-tuned models

🛠️ Tech Stack
Component	Technology
Frontend	Flutter / React / HTML/CSS/JS (choose your project)
Backend	Node.js / Python FastAPI / Flask
Database	MongoDB / Firebase / PostgreSQL
AI Engine	OpenAI, Llama 3, Gemma, Mistral, or custom
Speech-to-Text	Google STT / Whisper
Authentication	Firebase Auth / JWT
📂 Project Structure
AI-Mock-Interview/
│
├── backend/
│   ├── app.py / server.js
│   ├── routes/
│   ├── controllers/
│   └── models/
│
├── frontend/
│   ├── lib/ (flutter)
│   ├── src/ (react)
│   └── assets/
│
├── ai/
│   ├── prompt_templates/
│   ├── evaluation_model/
│   └── question_generator.py
│
├── docs/
│   └── architecture.md
│
└── README.md

⚙️ Installation & Setup
1. Clone the repository
git clone https://github.com/pran2208/interview-system-with-ai-ml
cd AI-Mock-Interview

2. Setup Backend
cd backend
pip install -r requirements.txt   # Python
# or
npm install                       # Node.js


Add your API keys in .env:

OPENAI_API_KEY=your_key
MONGO_URI=your_mongo_url
JWT_SECRET=your_secret

3. Run Backend
python app.py
# or
npm run start

4. Run Frontend

(Flutter Example)

flutter pub get
flutter run

🧪 How the System Works
1️⃣ User selects domain →
2️⃣ AI generates adaptive questions →
3️⃣ User answers via text/voice →
4️⃣ NLP engine evaluates response →
5️⃣ AI provides feedback & scoring →
6️⃣ Performance analytics are stored
🔥 Core AI Components
1. Interview Question Generator

Uses:

Domain prompts

Difficulty scaling

Company-specific patterns

2. Answer Evaluator

Evaluates based on:

Technical correctness

STAR method structure

Depth of explanation

Communication & tone

3. Feedback Engine

Generates:

Strengths

Weaknesses

Improved sample answer

Personalized recommendations

📈 Future Enhancements

Resume parser → auto-generate interview questions

Video interviews with facial analysis

Multi-language support

Leaderboard & competitive mode

HR ATS scoring integration

Offline interview practice using local LLMs

🤝 Contributing

Pull requests are welcome!
For major changes, please open an issue first to discuss what you’d like to modify.

📜 License

This project is licensed under the MIT License.

⭐ Support

If you like this project, give it a ⭐ on GitHub and share it with others!
