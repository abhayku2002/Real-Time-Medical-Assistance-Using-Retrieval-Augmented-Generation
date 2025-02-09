# 🩺 Real-Time Medical Diagnosis Assistant

An AI-powered medical assistant built using Streamlit, LangChain, and Ollama. This application provides real-time medical diagnosis assistance and a health risk calculator.

## 🚀 Features
- AI Medical Assistant powered by Ollama & LangChain.
- Health Risk Calculator based on user inputs.
- Clean and interactive UI using Streamlit.
- Customizable AI models for medical responses.

## 📌 Technologies Used
- **Streamlit** (Frontend and UI)
- **LangChain** (LLM framework)
- **Ollama** (Local AI model inference)

---

## 📥 Installation
### Prerequisites
Make sure you have **Python 3.8+** installed.

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/your-username/medical-ai-assistant.git
cd medical-ai-assistant
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```sh
streamlit run app.py
```

---

## ⚙️ Configuration
- Choose AI model: Select a model from the sidebar (`deepseek-r1:1.5b`, `deepseek-r1:7b`, `deepseek-r1:latest`).
- Switch between **Medical Assistant** and **Health Risk Calculator** modes.

## 🛠 How It Works
### Medical Assistant:
1. Enter a medical query in the chat input.
2. The AI assistant processes and provides a response based on LangChain & Ollama.

### Health Risk Calculator:
1. Enter details like age, weight, height, smoking habits, and exercise frequency.
2. Click **Calculate Risk** to receive an estimated health risk level.

---

## 📌 Requirements
Make sure **Ollama** is installed and running locally.

### Install Ollama (if not installed)
```sh
curl -fsSL https://ollama.ai/install.sh | sh
```

Start the Ollama server:
```sh
ollama serve
```

---

## 💡 Future Enhancements
- Add support for voice-based queries.
- Integration with real-world medical datasets.
- More accurate risk prediction models.

## 📝 License
This project is open-source under the **MIT License**.

## 👨‍💻 Authors
- **Abhay Kumar**
- **Prakash Kumar Nayak**
- **Ankit Raj Sharma**
- **Mudit Kumar Sharma**

