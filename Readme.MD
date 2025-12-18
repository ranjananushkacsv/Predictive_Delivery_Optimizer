# 🚚 Delivery Delay Predictor with AI Assistant

An end-to-end **Logistics Analytics & AI Application** built using **Streamlit**, **Machine Learning**, and a **Local LLM (Ollama)** to predict delivery delays, recommend optimal carriers, analyze cost impact, and interact via a natural language chatbot.

This project demonstrates how **Data Science + ML models + LLMs** can be combined to solve real-world supply chain problems.

---

## 📌 Key Features

### 📈 Interactive Dashboard
- Overall system metrics (accuracy, delay rate)
- Carrier on-time performance visualization
- Top delivery risk factors

### 🔮 Delivery Delay Prediction (ML-powered)
- Predicts probability of delivery delay
- Considers:
  - Carrier risk
  - Traffic delay
  - Weather conditions
  - Distance
  - Priority level
  - International shipments
  - Promised delivery days
- Provides **risk level classification** and **actionable recommendations**

### 🚚 Carrier Recommendation System
- Compares multiple logistics carriers
- Ranks carriers based on:
  - On-time probability
  - Delay risk
  - Estimated cost
  - Weighted performance score
- Helps choose the **best carrier for given constraints**

### 💰 Delay Cost Impact Analysis
- Calculates:
  - Direct delay cost
  - Customer churn risk
  - Total business impact
- Provides **ROI analysis** for preventive actions

### 🤖 AI-Powered Logistics Assistant
- Chat naturally using **Ollama (local LLM)**
- Understands queries like:
  - “What’s the delay risk for Mumbai to Delhi express delivery?”
  - “Which carrier is most reliable?”
  - “Calculate loss if my order is delayed”
- Uses:
  - Rule-based parameter extraction
  - ML-based / mock delay prediction
  - Context-aware LLM responses

---

## 🧠 Tech Stack

| Layer | Technology |
|-----|-----------|
| Frontend | Streamlit |
| Machine Learning | Scikit-learn, XGBoost |
| Visualization | Plotly |
| LLM | Ollama (Mistral / LLaMA2 / Mixtral) |
| Backend | Python |
| Model Storage | Joblib |

---

## 📂 Project Structure

