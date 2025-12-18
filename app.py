import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import requests
import json
import re

# Page config
st.set_page_config(
    page_title="Delivery Delay Predictor",
    page_icon="🚚",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    delay_model = joblib.load('model.pkl')
    features = open('features.txt').read().splitlines()
    carrier_risk = joblib.load('carrier_risk.pkl')
    mappings = joblib.load('mappings.pkl')
    carrier_model = joblib.load('carrier_model.pkl')
    carrier_perf = joblib.load('carrier_performance.pkl')
    return delay_model, features, carrier_risk, mappings, carrier_model, carrier_perf

try:
    delay_model, features, carrier_risk, mappings, carrier_model, carrier_perf = load_models()
except:
    st.error("Model files missing! Please upload: model.pkl, features.txt, carrier_risk.pkl, mappings.pkl, carrier_model.pkl, carrier_performance.pkl")
    st.stop()

# Ollama setup (for chatbot)
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"  # Change to your installed model

# Chatbot helper functions
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
            "temperature": 0.2,
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
    
    # Distance detection
    distance_match = re.search(r'(\d+)\s*(km|kilometers|kilometer)', query.lower())
    if distance_match:
        params['distance'] = int(distance_match.group(1))
    
    # Weather detection
    weather_terms = ["rain", "storm", "fog", "snow", "sunny"]
    for term in weather_terms:
        if term in query.lower():
            params['weather'] = term.capitalize()
            break
    
    return params

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
    
    # Distance adjustment
    if params.get('distance'):
        if params['distance'] > 1000:
            risk += 10
        elif params['distance'] > 500:
            risk += 5
    
    # Clamp between 5-95%
    risk = max(5, min(95, risk))
    
    return risk

def process_logistics_query(query):
    """Process query and generate intelligent response"""
    params = extract_logistics_params(query)
    
    # Prepare context for Ollama
    context = f"""
    USER QUERY PARAMETERS EXTRACTED:
    {json.dumps(params, indent=2)}
    
    PREDICTION CONTEXT:
    """
    
    # Add prediction if we have parameters
    if params:
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

# App header
st.title("🚚 Delivery Delay Predictor")
st.markdown("Predict delivery delays and optimize logistics operations")

# Sidebar - UPDATE THIS TO ADD CHATBOT OPTION
st.sidebar.header("📊 Navigation")
page = st.sidebar.radio("Go to:", ["📈 Dashboard", "🔮 Predict Delay", "🚚 Carrier Recommender", "💰 Cost Analysis", "🤖 AI Assistant"])

# Page 1: Dashboard
if page == "📈 Dashboard":
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Models", "2", "Active")
    with col2:
        st.metric("Accuracy", "83.5%", "+2.3%")
    with col3:
        st.metric("Avg Delay Rate", "24.7%", "-1.2%")
    
    st.subheader("📊 Performance Overview")
    
    # Carrier performance chart
    carriers = list(carrier_perf.keys())
    on_time_rate = [(1 - carrier_perf[c])*100 for c in carriers]
    
    fig = go.Figure(data=[
        go.Bar(name='On-Time %', x=carriers, y=on_time_rate, marker_color='lightgreen')
    ])
    fig.update_layout(title='Carrier On-Time Performance', yaxis_title='On-Time %')
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk factors
    st.subheader("⚠️ Top Risk Factors")
    risk_factors = {
        'Heavy Rain': 78,
        'Traffic > 60min': 65,
        'SpeedyLogistics': 52,
        'International': 48,
        'Economy Priority': 45
    }
    
    for factor, risk in risk_factors.items():
        st.progress(risk/100, text=f"{factor}: {risk}% risk")

# Page 2: Predict Delay
elif page == "🔮 Predict Delay":
    st.header("🔮 Predict Delivery Delay")
    
    col1, col2 = st.columns(2)
    
    with col1:
        carrier = st.selectbox("Carrier", list(carrier_risk.keys()))
        traffic = st.slider("Traffic Delay (minutes)", 0, 180, 30)
        weather = st.selectbox("Weather", list(mappings['weather_score'].keys()))
        distance = st.number_input("Distance (km)", 10, 5000, 500)
    
    with col2:
        priority = st.selectbox("Priority", list(mappings['priority_score'].keys()))
        promised_days = st.number_input("Promised Days", 1, 30, 3)
        international = st.selectbox("Destination Type", ["Domestic", "International"])
        
        if st.button("🔮 Predict Delay", type="primary"):
            # Prepare input
            input_data = {
                'carrier_risk': carrier_risk[carrier],
                'traffic_risk': traffic / 100,
                'weather_risk': mappings['weather_score'][weather],
                'distance_risk': min(distance / 2000, 1),
                'priority_risk': mappings['priority_score'][priority],
                'international': 1 if international == "International" else 0,
                'promised_risk': 1 / (promised_days + 1)
            }
            
            # Create DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Ensure all features present
            for feature in features:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            
            # Predict
            prob = delay_model.predict_proba(input_df[features])[0][1] * 100
            
            # Display result
            st.subheader("🎯 Prediction Result")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Delay Probability", f"{prob:.1f}%")
            with col_b:
                risk_level = "HIGH" if prob > 70 else "MEDIUM" if prob > 30 else "LOW"
                st.metric("Risk Level", risk_level)
            with col_c:
                prediction = "🚨 DELAYED" if prob > 50 else "✅ ON-TIME"
                st.metric("Prediction", prediction)
            
            # Recommendations
            st.subheader("💡 Recommendations")
            if prob > 70:
                st.error("""
                🚨 **Critical Risk - Take Immediate Action:**
                - Switch to QuickShip or ReliableExpress
                - Add 2-day buffer to promised date
                - Assign backup vehicle
                - Notify customer in advance
                """)
            elif prob > 50:
                st.warning("""
                ⚠️ **Medium Risk - Consider Actions:**
                - Add 1-day buffer
                - Monitor traffic updates
                - Prepare contingency plan
                """)
            else:
                st.success("""
                ✅ **Low Risk - Proceed Normally:**
                - Standard protocol is sufficient
                - Monitor for unexpected issues
                """)

# Page 3: Carrier Recommender
elif page == "🚚 Carrier Recommender":
    st.header("🚚 Best Carrier Recommender")
    
    col1, col2 = st.columns(2)
    
    with col1:
        distance = st.number_input("Delivery Distance (km)", 10, 5000, 300)
        priority_rec = st.selectbox("Delivery Priority", ["Express", "Standard", "Economy"])
        weather_rec = st.selectbox("Weather Forecast", ["None", "Light Rain", "Heavy Rain", "Fog"])
    
    with col2:
        traffic_rec = st.slider("Expected Traffic (minutes)", 0, 120, 20)
        is_international_rec = st.selectbox("Is International?", ["No", "Yes"])
        
        if st.button("🎯 Find Best Carrier", type="primary"):
            # Define the correct feature order as expected by the model
            correct_feature_order = [
                'carrier_delay_rate', 
                'promised_days', 
                'carrier_Ecodeliver', 
                'carrier_Globaltransit', 
                'carrier_Quickship', 
                'carrier_Reliableexpress', 
                'carrier_Speedylogistics'
            ]
            
            # Calculate scores for each carrier
            results = []
            for carrier in carrier_perf.keys():
                # Prepare features
                input_data = {
                    'carrier_delay_rate': carrier_perf[carrier],
                    'promised_days': 3,
                    'carrier_Speedylogistics': 1 if carrier == 'Speedylogistics' else 0,
                    'carrier_Quickship': 1 if carrier == 'Quickship' else 0,
                    'carrier_Globaltransit': 1 if carrier == 'Globaltransit' else 0,
                    'carrier_Reliableexpress': 1 if carrier == 'Reliableexpress' else 0,
                    'carrier_Ecodeliver': 1 if carrier == 'Ecodeliver' else 0
                }
                
                # Create DataFrame and reorder columns to match model's expected order
                input_df = pd.DataFrame([input_data])
                
                # Ensure all columns are present and in correct order
                for col in correct_feature_order:
                    if col not in input_df.columns:
                        input_df[col] = 0
                
                # Reorder columns to match model's expected order
                input_df = input_df[correct_feature_order]
                
                # Predict
                delay_prob = carrier_model.predict_proba(input_df)[0][1] * 100
                on_time_prob = 100 - delay_prob
                
                # Cost factor
                base_cost = 500
                if carrier == 'Ecodeliver':
                    cost = base_cost * 0.9
                elif carrier == 'Speedylogistics':
                    cost = base_cost * 0.8
                else:
                    cost = base_cost
                
                results.append({
                    'Carrier': carrier,
                    'On-Time %': on_time_prob,
                    'Delay Risk %': delay_prob,
                    'Est. Cost (₹)': cost,
                    'Score': on_time_prob * 0.7 - cost/10 * 0.3
                })
            
            # Sort by score
            results_df = pd.DataFrame(results).sort_values('Score', ascending=False)
            
            st.subheader("🏆 Recommended Carriers")
            
            # Display as metrics
            top3 = results_df.head(3)
            cols = st.columns(3)
            for idx, (_, row) in enumerate(top3.iterrows()):
                with cols[idx]:
                    st.metric(
                        label=f"#{idx+1} {row['Carrier']}",
                        value=f"{row['On-Time %']:.1f}%",
                        delta=f"₹{row['Est. Cost (₹)']:.0f}"
                    )
            
            # Detailed table
            st.subheader("📋 All Carriers Comparison")
            st.dataframe(results_df.style.format({
                'On-Time %': '{:.1f}%',
                'Delay Risk %': '{:.1f}%',
                'Est. Cost (₹)': '₹{:.0f}',
                'Score': '{:.1f}'
            }).highlight_max(subset=['Score'], color='lightgreen'), use_container_width=True)

# Page 4: Cost Analysis
elif page == "💰 Cost Analysis":
    st.header("💰 Delay Cost Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        order_value = st.number_input("Order Value (₹)", 100, 1000000, 5000)
        delay_days = st.slider("Expected Delay (days)", 0, 15, 2)
        priority_cost = st.selectbox("Priority", ["Express", "Standard", "Economy"])
    
    with col2:
        customer_type = st.selectbox("Customer Type", ["Individual", "SMB", "Enterprise"])
        is_international_cost = st.selectbox("International Shipment", ["No", "Yes"])
        
        if st.button("💰 Calculate Impact", type="primary"):
            # Cost calculations
            base_cost = 850  # From data analysis
            
            # Multiply factors
            cost_factors = {
                'Express': 1.5,
                'Standard': 1.0,
                'Economy': 0.8
            }
            
            customer_factors = {
                'Individual': 1.0,
                'SMB': 1.2,
                'Enterprise': 1.5
            }
            
            international_factor = 1.3 if is_international_cost == "Yes" else 1.0
            
            # Calculate
            delay_cost = base_cost * delay_days * cost_factors[priority_cost] * customer_factors[customer_type] * international_factor
            
            # Customer churn cost
            churn_risk = min(30 + (delay_days * 10), 80)  # 30% base + 10% per day
            churn_cost = order_value * (churn_risk / 100)
            
            total_cost = delay_cost + churn_cost
            
            # Display results
            st.subheader("📊 Cost Breakdown")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Direct Delay Cost", f"₹{delay_cost:,.0f}")
            with col_b:
                st.metric("Customer Churn Risk", f"{churn_risk:.0f}%", f"₹{churn_cost:,.0f}")
            with col_c:
                st.metric("Total Impact", f"₹{total_cost:,.0f}", delta="Loss")
            
            # ROI of prevention
            st.subheader("📈 Prevention ROI")
            prevention_cost = delay_cost * 0.3  # 30% of delay cost
            roi = ((total_cost - prevention_cost) / prevention_cost) * 100
            
            st.info(f"""
            **Invest ₹{prevention_cost:,.0f} in prevention measures to save ₹{total_cost:,.0f}**
            
            📊 **ROI: {roi:.0f}%**
            
            ✅ **Recommended Actions:**
            - Real-time tracking: ₹15,000
            - Weather alerts: ₹5,000  
            - Buffer vehicles: ₹25,000
            - Customer notifications: ₹2,000
            """)

# Page 5: AI Assistant (New Chatbot Page) - ADD THIS NEW SECTION
elif page == "🤖 AI Assistant":
    st.header("🤖 Logistics AI Assistant")
    st.markdown("Chat naturally about deliveries, delays, carriers, and costs")
    
    # Initialize chat history in session state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Hello! I'm your logistics assistant. I can help you with:\n\n1. 📦 Predict delivery delays\n2. 🚚 Recommend best carriers\n3. 💰 Calculate delay costs\n4. 📊 Analyze performance data\n\nWhat would you like to know?"}
        ]
    
    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Sidebar with quick actions for chatbot
    with st.sidebar:
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
            if st.button(example, key=f"ex_{example[:10]}"):
                # Add to chat
                st.session_state.chat_messages.append({"role": "user", "content": example})
                with st.chat_message("user"):
                    st.markdown(example)
                
                # Get response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = process_logistics_query(example)
                        st.markdown(response)
                
                st.session_state.chat_messages.append({"role": "assistant", "content": response})
                st.rerun()
        
        st.divider()
        
        if st.button("🔄 Clear Chat History", type="secondary"):
            st.session_state.chat_messages = [
                {"role": "assistant", "content": "Chat cleared! How can I help you today?"}
            ]
            st.rerun()
        
        # Ollama status
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            if resp.status_code == 200:
                st.success("✅ Ollama Connected")
            else:
                st.warning("⚠️ Ollama Issues")
        except:
            st.error("❌ Ollama Not Running")
            st.caption("Run: `ollama serve` in terminal")
    
    # Chat input
    if prompt := st.chat_input("Ask about deliveries, delays, carriers..."):
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your logistics query..."):
                response = process_logistics_query(prompt)
                st.markdown(response)
        
        # Add assistant response
        st.session_state.chat_messages.append({"role": "assistant", "content": response})

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**Delivery Delay Predictor v2.0**  
Predict delays, optimize carriers, reduce costs, and chat with AI
""")