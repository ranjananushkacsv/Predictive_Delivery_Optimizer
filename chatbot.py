# chatbot.py
import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Logistics AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# Ollama setup
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"  # Change to llama2 or mixtral if you have

# Load your ML models (if available)
try:
    import joblib
    delay_model = joblib.load('model.pkl')
    carrier_perf = joblib.load('carrier_performance.pkl')
    ml_loaded = True
except:
    ml_loaded = False
    st.warning("ML models not loaded. Chatbot will work but can't give precise predictions.")

# App title
st.title("🤖 Logistics AI Assistant")
st.markdown("Chat naturally about deliveries, delays, carriers, and costs")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your logistics assistant. I can help you with:\n\n1. 📦 Predict delivery delays\n2. 🚚 Recommend best carriers\n3. 💰 Calculate delay costs\n4. 📊 Analyze performance data\n\nWhat would you like to know?"}
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Ollama response function
def get_ollama_response(prompt, context=""):
    """Get response from Ollama"""
    
    full_prompt = f"""
    You are a Logistics and Supply Chain Expert Assistant.
    You help users with delivery delay predictions, carrier recommendations, and cost analysis.
    
    CONTEXT FOR THIS CONVERSATION:
    {context}
    
    IMPORTANT DATA (use when relevant):
    - Carriers available: SpeedyLogistics, QuickShip, GlobalTransit, ReliableExpress, EcoDeliver
    - QuickShip has 73% on-time rate (best)
    - SpeedyLogistics has 48% on-time rate (worst)
    - Base delay cost: ₹850 per delayed delivery
    - Express priority has 15% delay rate
    - Economy priority has 45% delay rate
    - International deliveries have 40% higher delay risk
    
    RESPONSE GUIDELINES:
    1. Be concise and helpful
    2. Use bullet points for lists
    3. Include emojis for better readability
    4. Always suggest actionable steps
    5. If user asks for prediction but doesn't give details, ask clarifying questions
    
    USER QUERY: {prompt}
    
    ASSISTANT RESPONSE:
    """
    
    payload = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,  # Low for factual responses
            "top_p": 0.9,
            "num_predict": 500
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return "I apologize, but I'm having trouble connecting to my knowledge base. Please try again or use the manual prediction forms."
    except requests.exceptions.ConnectionError:
        return "⚠️ **Ollama is not running!**\n\nTo start the chatbot:\n```bash\nollama serve\n```\nThen refresh this page."
    except Exception as e:
        return f"Error: {str(e)}"

# Extract logistics parameters from query
def extract_logistics_params(query):
    """Extract logistics parameters from natural language query"""
    params = {}
    
    # Carrier detection
    carriers = ["speedy", "quickship", "global", "reliable", "eco"]
    for carrier in carriers:
        if carrier in query.lower():
            params['carrier'] = carrier.capitalize() + ("Logistics" if carrier == "speedy" else "Express" if carrier == "reliable" else "Transit" if carrier == "global" else "Deliver" if carrier == "eco" else "Ship")
            break
    
    # Priority detection
    if "express" in query.lower():
        params['priority'] = "Express"
    elif "economy" in query.lower():
        params['priority'] = "Economy"
    else:
        params['priority'] = "Standard"
    
    # City detection (basic)
    cities = ["mumbai", "delhi", "bangalore", "chennai", "kolkata", "hyderabad", "pune", "ahmedabad"]
    for city in cities:
        if city in query.lower():
            if 'origin' not in params:
                params['origin'] = city.title()
            else:
                params['destination'] = city.title()
    
    # International detection
    intl_dest = ["dubai", "singapore", "hong kong", "bangkok"]
    for dest in intl_dest:
        if dest in query.lower():
            params['destination'] = dest.title()
            params['international'] = True
    
    return params

# Generate mock prediction (if ML models not available)
def mock_prediction(params):
    """Generate realistic mock prediction based on parameters"""
    
    base_risk = 25  # Base 25% risk
    
    # Carrier adjustments
    carrier_adj = {
        "SpeedyLogistics": +30,
        "QuickShip": -10,
        "ReliableExpress": -8,
        "GlobalTransit": +5,
        "EcoDeliver": +15
    }
    
    # Priority adjustments
    priority_adj = {"Express": -10, "Standard": 0, "Economy": +20}
    
    # Calculate
    risk = base_risk
    if params.get('carrier') in carrier_adj:
        risk += carrier_adj[params['carrier']]
    
    if params.get('priority') in priority_adj:
        risk += priority_adj[params['priority']]
    
    if params.get('international'):
        risk += 15
    
    # Clamp between 5-95%
    risk = max(5, min(95, risk))
    
    return risk

# Process user query intelligently
def process_logistics_query(query):
    """Process query and generate intelligent response"""
    
    # Extract parameters
    params = extract_logistics_params(query)
    
    # Prepare context for Ollama
    context = f"""
    USER QUERY PARAMETERS EXTRACTED:
    {json.dumps(params, indent=2)}
    
    PREDICTION CONTEXT:
    """
    
    # Add prediction if we have parameters
    if params:
        if ml_loaded:
            # Use actual ML model
            # This is simplified - you'd need to format input properly
            pass
        else:
            # Use mock prediction
            predicted_risk = mock_prediction(params)
            context += f"Estimated delay risk: {predicted_risk}%\n"
            
            # Recommendations
            if predicted_risk > 60:
                context += "RECOMMENDATION: High risk! Consider changing carrier to QuickShip, add buffer days, and notify customer.\n"
            elif predicted_risk > 30:
                context += "RECOMMENDATION: Moderate risk. Monitor closely and have contingency plan.\n"
            else:
                context += "RECOMMENDATION: Low risk. Proceed as planned.\n"
    
    # Get response from Ollama
    return get_ollama_response(query, context)

# Sidebar with quick actions
with st.sidebar:
    st.header("⚡ Quick Actions")
    
    if st.button("🔄 Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat cleared! How can I help you today?"}
        ]
        st.rerun()
    
    st.divider()
    
    st.subheader("💡 Example Questions")
    examples = [
        "What's the delay risk for Mumbai to Delhi express delivery?",
        "Which carrier is best for fragile items?",
        "Calculate cost if my ₹20,000 order is delayed 2 days",
        "Why does SpeedyLogistics have so many delays?",
        "Compare QuickShip vs ReliableExpress",
        "How does weather affect delivery times?"
    ]
    
    for example in examples:
        if st.button(example, key=example):
            st.session_state.messages.append({"role": "user", "content": example})
            with st.chat_message("user"):
                st.markdown(example)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = process_logistics_query(example)
                    st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    
    st.divider()
    
    # Ollama status
    st.subheader("🔧 System Status")
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        if resp.status_code == 200:
            st.success("✅ Ollama Connected")
            models = resp.json().get("models", [])
            if models:
                st.caption(f"Models: {', '.join([m['name'] for m in models])}")
    except:
        st.error("❌ Ollama Not Running")
        st.caption("Start with: `ollama serve`")

# Main chat input
if prompt := st.chat_input("Ask about deliveries, delays, carriers..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your logistics query..."):
            response = process_logistics_query(prompt)
            st.markdown(response)
    
    # Add assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})