"""
Simplified Streamlit Pitch Deck Generator
Basic version without CrewAI for Streamlit Cloud compatibility.
"""

import json
import os
import logging
import traceback
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
import openai
from rag_implementation import EnhancedRAGAgent
from simple_rag import SimpleRAG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Streamlit Configuration
st.set_page_config(
    page_title="🚀 AI Pitch Deck Generator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to match the original design
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #4f46e5;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .workflow-info {
        background-color: #eff6ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background-color: #4f46e5;
        color: white;
        font-weight: bold;
        padding: 0.75rem 2rem;
        border-radius: 0.5rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #4338ca;
        transform: scale(1.05);
    }
    .footer-info {
        background-color: #f9fafb;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin-top: 2rem;
        font-size: 0.75rem;
        color: #6b7280;
    }
</style>
""", unsafe_allow_html=True)

def configure_openai():
    """Configure OpenAI client."""
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
        if not openai_api_key:
            st.error("OpenAI API key not found. Please add it to your Streamlit secrets.")
            st.stop()

        openai.api_key = openai_api_key
        return True
    except Exception as e:
        st.error(f"Failed to configure OpenAI: {str(e)}")
        return False

def initialize_rag():
    """Initialize RAG system."""
    try:
        rag_system = SimpleRAG()
        logger.info("RAG system initialized successfully")
        return rag_system
    except Exception as e:
        logger.error(f"Failed to initialize RAG: {str(e)}")
        st.error(f"Failed to initialize RAG system: {str(e)}")
        return None

def initialize_rag():
    """Initialize RAG system."""
    try:
        rag_system = SimpleRAG()
        return rag_system
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        return None

def generate_pitch_content_with_rag(startup_data, rag_system):
    """Generate pitch deck content using OpenAI with RAG analysis."""
    try:
        # First, perform RAG analysis
        rag_analysis = rag_system.analyze_with_rag(startup_data)

        # Check if company exists in database
        company_check = rag_system.check_company_exists(startup_data['startup_name'])

        prompt = f"""
        Create a comprehensive pitch deck for the startup "{startup_data['startup_name']}" using the following information and RAG analysis:

        STARTUP INFORMATION:
        Company: {startup_data['startup_name']}
        Industry: {startup_data['industry_type']}
        Founder: {startup_data['founder_name']}
        Product: {startup_data['product_name']}

        Vision: {startup_data['vision_statement']}
        Problem: {startup_data['key_problem_solved']}
        Solution: {startup_data['solution_summary']}
        Target Customer: {startup_data['target_customer_profile']}
        Business Model: {startup_data['business_model']}
        Market Size: {startup_data['market_size']}
        Competitors: {startup_data['competitors']}
        Competitive Advantage: {startup_data['why_you_win']}
        Funding Amount: {startup_data['funding_amount']}
        Use of Funds: {startup_data['use_of_funds_split_percentages']}
        Monetization: {startup_data['monetization_plan']}

        RAG ANALYSIS FROM DATABASE:
        {rag_analysis}

        COMPANY DATABASE STATUS:
        {"✅ Company found in database" if company_check['exists'] else "🆕 New company - not in database"}

        Create a detailed pitch deck incorporating insights from the RAG analysis:
        1. Title Slide
        2. Problem Statement (enhanced with market insights)
        3. Solution (positioned against similar companies)
        4. Market Opportunity (validated with database patterns)
        5. Business Model (compared to successful examples)
        6. Competitive Analysis (informed by database)
        7. Marketing Strategy (based on successful patterns)
        8. Financial Projections (benchmarked against similar companies)
        9. Funding Requirements (aligned with market standards)
        10. Team

        Include specific references to insights from similar companies in the database.
        Format the response as a structured pitch deck with clear sections and compelling content.
        """

        from openai import OpenAI
        client = OpenAI(api_key=openai.api_key)

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert pitch deck consultant with access to a database of successful pitch decks. Use the RAG analysis to create data-driven presentations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=3000,
            temperature=0.7
        )

        return response.choices[0].message.content, rag_analysis

    except Exception as e:
        logger.error(f"Error generating pitch content with RAG: {e}")
        return f"Error generating content: {str(e)}", ""

def display_main_form():
    """Display the main form interface matching the original design."""
    
    # Header section
    st.markdown('<h1 class="main-header">🚀 AI Pitch Deck Generator with RAG</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by OpenAI GPT-4 + RAG Technology</p>', unsafe_allow_html=True)

    # Workflow info box
    st.markdown("""
    <div class="workflow-info">
        <strong>🧠 RAG Workflow:</strong> Database Search → Similar Companies Analysis → AI-Enhanced Content Generation
    </div>
    """, unsafe_allow_html=True)
    
    # Form
    with st.form("pitch_deck_form"):
        # Row 1 - Basic Info
        col1, col2 = st.columns(2)
        with col1:
            startup_name = st.text_input("Startup Name", key="startup_name")
        with col2:
            industry_type = st.text_input("Industry Type", key="industry_type")
        
        # Row 2 - Founder and Product
        col1, col2 = st.columns(2)
        with col1:
            founder_name = st.text_input("Founder Name", key="founder_name")
        with col2:
            product_name = st.text_input("Product Name", key="product_name")
        
        # Text areas
        founder_bio = st.text_area("Founder Bio", height=80, key="founder_bio")
        team_summary = st.text_area("Team Summary", height=80, key="team_summary")
        vision_statement = st.text_area("Vision Statement", height=80, key="vision_statement")
        key_problem_solved = st.text_area("Key Problem You Solve", height=80, key="key_problem_solved")
        solution_summary = st.text_area("Solution Summary (1-3 lines)", height=80, key="solution_summary")
        target_customer_profile = st.text_area("Target Customer Profile", height=80, key="target_customer_profile")
        business_model = st.text_area("Business Model", height=80, key="business_model")
        acquisition_strategy = st.text_area("Acquisition Strategy", height=80, key="acquisition_strategy")
        
        # Single line inputs
        market_size = st.text_input("Market Size", key="market_size")
        competitors = st.text_input("Competitors (comma-separated)", key="competitors")
        
        # More text areas
        why_you_win = st.text_area("Why You Win", height=80, key="why_you_win")
        
        # Financial inputs
        funding_amount = st.text_input("Funding Amount", key="funding_amount")
        use_of_funds_split_percentages = st.text_input("Use of Funds (split into %)", key="use_of_funds_split_percentages")
        transactions = st.text_input("Transactions (if any)", key="transactions")
        monetization_plan = st.text_area("Monetization Plan", height=80, key="monetization_plan")
        
        # Submit button
        submitted = st.form_submit_button("🚀 Generate AI Pitch Deck", use_container_width=True)
        
        if submitted:
            # Validate required fields
            required_fields = [startup_name, industry_type, founder_name, product_name, 
                             founder_bio, team_summary, vision_statement, key_problem_solved,
                             solution_summary, target_customer_profile, business_model,
                             acquisition_strategy, market_size, why_you_win, funding_amount,
                             use_of_funds_split_percentages, monetization_plan]
            
            if all(field.strip() for field in required_fields):
                # Store form data in session state
                st.session_state.startup_data = {
                    'startup_name': startup_name,
                    'industry_type': industry_type,
                    'founder_name': founder_name,
                    'founder_bio': founder_bio,
                    'team_summary': team_summary,
                    'product_name': product_name,
                    'vision_statement': vision_statement,
                    'key_problem_solved': key_problem_solved,
                    'solution_summary': solution_summary,
                    'target_customer_profile': target_customer_profile,
                    'business_model': business_model,
                    'acquisition_strategy': acquisition_strategy,
                    'market_size': market_size,
                    'competitors': competitors,
                    'why_you_win': why_you_win,
                    'funding_amount': funding_amount,
                    'use_of_funds_split_percentages': use_of_funds_split_percentages,
                    'transactions': transactions,
                    'monetization_plan': monetization_plan
                }
                
                # Set processing state
                st.session_state.processing = True
                st.rerun()
            else:
                st.error("Please fill in all required fields.")
    
    # Footer
    st.markdown("""
    <div class="footer-info">
        🤖 Powered by OpenAI GPT-4 | 🔍 RAG Technology | 📊 Database of Successful Pitch Decks
    </div>
    """, unsafe_allow_html=True)

def display_processing():
    """Display processing status."""
    st.markdown('<h1 class="main-header">🔄 Processing Your Request</h1>', unsafe_allow_html=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # RAG-enhanced progress updates
    steps = [
        "🔍 Searching pitch deck database for similar companies...",
        "📊 Analyzing patterns from successful startups...",
        "🧠 Performing RAG analysis with historical data...",
        "📝 Generating AI-enhanced pitch deck content...",
        "✅ RAG-powered pitch deck complete!"
    ]
    
    for i, step in enumerate(steps):
        status_text.text(step)
        progress_bar.progress((i + 1) / len(steps))
        
        if i == 2:  # Generate content at step 3
            if 'pitch_content' not in st.session_state:
                with st.spinner("Generating your pitch deck with RAG analysis..."):
                    rag_system = st.session_state.get('rag_system')
                    if rag_system:
                        content, rag_analysis = generate_pitch_content_with_rag(st.session_state.startup_data, rag_system)
                        st.session_state.pitch_content = content
                        st.session_state.rag_analysis = rag_analysis
                    else:
                        st.session_state.pitch_content = "Error: RAG system not initialized"
        
        import time
        time.sleep(1)
    
    # Mark as complete
    st.session_state.processing = False
    st.session_state.show_results = True
    st.rerun()

def display_results():
    """Display results."""
    st.markdown('<h1 class="main-header">🎯 Your AI-Generated Pitch Deck</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Professional pitch deck content ready for presentation</p>', unsafe_allow_html=True)
    
    # New Analysis button
    if st.button("🔄 Create New Pitch Deck", type="primary"):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Display the generated content
    if 'pitch_content' in st.session_state:
        # Show RAG analysis first
        if 'rag_analysis' in st.session_state:
            st.markdown("## 🔍 RAG Analysis Results")
            with st.expander("📊 Database Search & Similar Companies Analysis", expanded=False):
                st.markdown(st.session_state.rag_analysis)

        st.markdown("## 📋 AI-Generated Pitch Deck Content")
        st.markdown(st.session_state.pitch_content)

        # Company info
        if 'startup_data' in st.session_state:
            st.markdown("---")
            st.markdown("### 📊 Company Information")
            data = st.session_state.startup_data
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Company:** {data['startup_name']}")
                st.info(f"**Industry:** {data['industry_type']}")
                st.info(f"**Founder:** {data['founder_name']}")
            with col2:
                st.info(f"**Product:** {data['product_name']}")
                st.info(f"**Funding:** {data['funding_amount']}")
                st.info(f"**Market Size:** {data['market_size']}")

    st.markdown("---")
    st.success("🎉 Your RAG-enhanced pitch deck has been generated! The content above includes insights from similar successful companies in our database.")

def main():
    """Main Streamlit application."""
    # Initialize session state
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False

    # Configure OpenAI
    if not configure_openai():
        st.stop()

    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        rag_system = initialize_rag()
        if rag_system:
            st.session_state.rag_system = rag_system
        else:
            st.error("Failed to initialize RAG system")
            st.stop()

    # Display appropriate page based on state
    if st.session_state.processing:
        display_processing()
    elif st.session_state.show_results:
        display_results()
    else:
        display_main_form()

if __name__ == "__main__":
    main()
