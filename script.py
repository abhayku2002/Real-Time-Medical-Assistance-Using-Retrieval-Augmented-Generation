import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

# Set page configuration
st.set_page_config(page_title="AI Medical Assistant", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        .message-container {
            padding: 15px;
            border-radius: 15px;
            margin-bottom: 10px;
            font-size: 16px;
            max-width: 70%;
        }
        .user-message {
            background-color: #d4edda;
            color: #155724;
            align-self: flex-end;
        }
        .ai-message {
            background-color: #cce5ff;
            color: #004085;
            align-self: flex-start;
        }
        .stButton button {
            background-color: #007bff;
            color: white;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Title and caption
st.title("🩺 Real Time Medical Diagnosis Assistant")
st.caption("Your AI-powered medical consultant")

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "Medical Assistant"

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    st.divider()
    
    # RAG Model Selection (for interface purposes only)
    st.subheader("RAG Model Selection")
    rag_model = st.radio(
        "Choose RAG Model",
        ["RAG 1 Model", "RAG 2 GB Model", "RAG 3 GB Model"],
        index=0
    )
    st.divider()
    
    # Page Navigation
    if st.button("Medical Assistant"):
        st.session_state.page = "Medical Assistant"
    if st.button("Health Risk Calculator"):
        st.session_state.page = "Health Risk Calculator"
    if st.button("Cancer Risk Assessment"):
        st.session_state.page = "Cancer Risk Assessment"
    
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")
    st.markdown("👨‍💻 Made By: Abhay Kumar, Prakash Kumar Nayak, Ankit Raj Sharma, Mudit Kumar Sharma")

# Initialize RAG model with the hardcoded model name
selected_model = "deepseek-r1:1.5b"  # Only this model is used in the backend
embeddings = OllamaEmbeddings(base_url="http://127.0.0.1:11434", model=selected_model)
vectorstore = FAISS.from_texts(["Medical knowledge base placeholder"], embeddings)

# Function to calculate cancer risk score
def calculate_cancer_risk_score(age, gender, family_history, smoking, alcohol, diet, physical_activity, environmental_exposure, medical_history):
    risk_score = 0

    # Age factor
    if age > 50:
        risk_score += 2
    elif age > 30:
        risk_score += 1

    # Gender-specific risk
    if gender == 'female':
        risk_score += 1  # Example: Breast cancer risk
    elif gender == 'male':
        risk_score += 1  # Example: Prostate cancer risk

    # Family history
    if family_history:
        risk_score += 2

    # Smoking
    if smoking:
        risk_score += 2

    # Alcohol consumption
    if alcohol:
        risk_score += 1

    # Diet
    if diet == 'unhealthy':
        risk_score += 1

    # Physical activity
    if not physical_activity:
        risk_score += 1

    # Environmental exposure
    if environmental_exposure:
        risk_score += 1

    # Medical history
    if medical_history:
        risk_score += 2

    return risk_score

# Cancer Risk Assessment Page
if st.session_state.page == "Cancer Risk Assessment":
    st.header("🩺 Cancer Risk Assessment")
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # Collect user inputs
    age = st.number_input("Enter your age:", min_value=0, max_value=120, step=1)
    gender = st.radio("Select your gender:", ["Male", "Female"])
    family_history = st.selectbox("Do you have a family history of cancer?", ["No", "Yes"])
    smoking = st.selectbox("Do you smoke?", ["No", "Yes"])
    alcohol = st.selectbox("Do you consume alcohol regularly?", ["No", "Yes"])
    diet = st.selectbox("Would you describe your diet as healthy?", ["Yes", "No"])
    physical_activity = st.selectbox("Do you engage in regular physical activity?", ["Yes", "No"])
    environmental_exposure = st.selectbox("Are you exposed to environmental carcinogens?", ["No", "Yes"])
    medical_history = st.selectbox("Do you have a personal medical history of cancer or related conditions?", ["No", "Yes"])

    if st.button("Assess Cancer Risk"):
        # Convert inputs to appropriate format
        gender = gender.lower()
        family_history = family_history == "Yes"
        smoking = smoking == "Yes"
        alcohol = alcohol == "Yes"
        diet = "healthy" if diet == "Yes" else "unhealthy"
        physical_activity = physical_activity == "Yes"
        environmental_exposure = environmental_exposure == "Yes"
        medical_history = medical_history == "Yes"

        # Calculate risk score
        risk_score = calculate_cancer_risk_score(age, gender, family_history, smoking, alcohol, diet, physical_activity, environmental_exposure, medical_history)

        # Determine risk level and percentage
        if risk_score >= 8:
            risk_level = "High"
            risk_percentage = "70-90%"
        elif risk_score >= 4:
            risk_level = "Moderate"
            risk_percentage = "30-70%"
        else:
            risk_level = "Low"
            risk_percentage = "10-30%"

        # Display the result
        st.markdown(f'<div class="message-container ai-message">Your estimated cancer risk level is: <strong>{risk_level}</strong></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="message-container ai-message">Risk Percentage: <strong>{risk_percentage}</strong></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="message-container ai-message">Note: This assessment provides a general risk estimation. For a comprehensive evaluation, please consult a healthcare professional.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Medical Assistant Page
if st.session_state.page == "Medical Assistant":
    llm_engine = ChatOllama(
        model=selected_model,
        base_url="http://127.0.0.1:11434",
        temperature=0.3
    )

    system_prompt = SystemMessagePromptTemplate.from_template(
        "You are an AI medical assistant. Provide concise and accurate medical information in English."
    )

    if "message_log" not in st.session_state:
        st.session_state.message_log = [{"role": "ai", "content": "Hello! I'm your AI Medical Assistant. How can I help?"}]

    chat_container = st.container()

    with chat_container:
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        for message in st.session_state.message_log:
            message_class = "user-message" if message["role"] == "user" else "ai-message"
            st.markdown(f'<div class="message-container {message_class}">{message["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    user_query = st.chat_input("Type your medical query here...")

    def generate_ai_response(prompt_chain):
        processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
        return processing_pipeline.invoke({})

    def build_prompt_chain():
        prompt_sequence = [system_prompt]
        for msg in st.session_state.message_log:
            if msg["role"] == "user":
                prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
            elif msg["role"] == "ai":
                prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
        return ChatPromptTemplate.from_messages(prompt_sequence)

    if user_query:
        st.session_state.message_log.append({"role": "user", "content": user_query})
        with st.spinner("🔍 Analyzing your medical query..."):
            # Retrieve relevant documents from the vector store
            docs = vectorstore.similarity_search(user_query, k=3)
            context = "\n".join([doc.page_content for doc in docs])
            prompt_chain = build_prompt_chain()
            ai_response = generate_ai_response(prompt_chain)
            ai_response = f"Context:\n{context}\n\nResponse:\n{ai_response}"
        st.session_state.message_log.append({"role": "ai", "content": ai_response})
        st.rerun()

# Health Risk Calculator Page
elif st.session_state.page == "Health Risk Calculator":
    st.header("🩺 Health Risk Calculator")
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    gender = st.radio("Select Gender", ["Male", "Female"])
    age = st.number_input("Enter your Age", min_value=0, max_value=120, step=1)
    weight = st.number_input("Enter your Weight (kg)", min_value=0.0, step=0.1)
    height = st.number_input("Enter your Height (cm)", min_value=0.0, step=0.1)
    smoking = st.selectbox("Do you smoke?", ["No", "Occasionally", "Regularly"])
    exercise = st.selectbox("How often do you exercise?", ["Rarely", "1-2 times a week", "3-5 times a week", "Daily"])
    blood_pressure = st.selectbox("Do you have high blood pressure?", ["No", "Yes"])
    diabetes = st.selectbox("Do you have diabetes?", ["No", "Yes"])

    if st.button("Calculate Risk"):
        bmi = weight / ((height / 100) ** 2)
        risk_score = sum([bmi > 30, smoking == "Regularly", exercise == "Rarely", age > 50, blood_pressure == "Yes", diabetes == "Yes"]) * 2
        risk_level = "Low" if risk_score <= 2 else "Moderate" if risk_score <= 4 else "High"
        st.write(f"Your estimated health risk level is: **{risk_level}**")
    st.markdown('</div>', unsafe_allow_html=True)
