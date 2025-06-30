"""
Test RAG Implementation with Beelinguapp Data
Verifies that RAG is working correctly with sample data similar to Beelinguapp.
"""

import os
import json
from dotenv import load_dotenv
from simple_rag import SimpleRAG

# Load environment variables
load_dotenv()

def test_rag_with_beelinguapp_data():
    """Test RAG implementation using Beelinguapp-like startup data."""
    
    print("🧪 Testing RAG Implementation with Beelinguapp-like Data")
    print("=" * 60)
    
    # Sample startup data similar to Beelinguapp
    beelinguapp_like_startup = {
        "startup_name": "LinguaStory",
        "industry_type": "EdTech",
        "founder_name": "Maria Rodriguez",
        "founder_bio": "Former language teacher with 10 years experience, passionate about making language learning engaging",
        "team_summary": "Team of 5: 2 developers, 1 content creator, 1 marketing specialist, 1 language expert",
        "product_name": "LinguaStory App",
        "vision_statement": "To make language learning as engaging as reading your favorite book",
        "key_problem_solved": "Traditional language learning apps are boring and don't provide real-world context",
        "solution_summary": "Interactive language learning through immersive stories with native audio narration",
        "target_customer_profile": "Young professionals aged 25-40 who want to learn languages for career advancement",
        "business_model": "Freemium model with premium subscriptions for advanced features",
        "acquisition_strategy": "Social media marketing, partnerships with educational institutions, referral programs",
        "market_size": "$60 billion global language learning market",
        "competitors": "Duolingo, Babbel, Rosetta Stone, Beelinguapp",
        "why_you_win": "Unique story-based approach with professional voice actors and adaptive difficulty",
        "funding_amount": "$500,000",
        "use_of_funds_split_percentages": "40% product development, 30% marketing, 20% team expansion, 10% operations",
        "transactions": "None yet - pre-revenue stage",
        "monetization_plan": "Premium subscriptions ($9.99/month), corporate partnerships, content licensing"
    }
    
    try:
        # Initialize RAG system
        print("🔧 Initializing RAG system...")
        rag_system = SimpleRAG()
        print("✅ RAG system initialized successfully")
        
        # Test 1: Check if Beelinguapp exists in database
        print("\n📋 Test 1: Checking if Beelinguapp exists in database...")
        company_check = rag_system.check_company_exists("Beelinguapp")
        print(f"Result: {company_check}")
        
        if company_check['exists']:
            print("✅ Beelinguapp found in database!")
            print(f"Company data: {json.dumps(company_check['company_data'], indent=2)}")
        else:
            print("❌ Beelinguapp not found in database")
        
        # Test 2: Find similar companies
        print("\n📋 Test 2: Finding similar companies to LinguaStory...")
        similar_companies = rag_system.find_similar_companies(beelinguapp_like_startup, top_k=3)
        print(f"Found {len(similar_companies)} similar companies:")
        
        for i, company in enumerate(similar_companies, 1):
            print(f"\n{i}. {company['company_name']} (Score: {company['relevance_score']})")
            print(f"   Industry: {company['industry']}")
            print(f"   Business Model: {company['business_model']}")
            print(f"   Competitive Advantage: {company['competitive_advantage']}")
        
        # Test 3: Full RAG analysis
        print("\n📋 Test 3: Performing full RAG analysis...")
        rag_analysis = rag_system.analyze_with_rag(beelinguapp_like_startup)
        
        print("RAG Analysis Result:")
        print("-" * 40)
        print(rag_analysis[:1000] + "..." if len(rag_analysis) > 1000 else rag_analysis)
        
        # Test 4: Database statistics
        print("\n📋 Test 4: Database statistics...")
        stats = rag_system.get_database_stats()
        print(f"Database stats: {json.dumps(stats, indent=2)}")
        
        # Test 5: Verify RAG components
        print("\n📋 Test 5: Verifying RAG components...")
        
        # Check if we can find EdTech companies
        edtech_companies = [company for company in similar_companies 
                           if 'edtech' in company['industry'].lower() or 'education' in company['industry'].lower()]
        
        print(f"✅ Found {len(edtech_companies)} EdTech companies in results")
        
        # Check if Beelinguapp is in the knowledge base
        beelinguapp_in_kb = 'beelinguapp' in rag_system.knowledge_base
        print(f"✅ Beelinguapp in knowledge base: {beelinguapp_in_kb}")
        
        # Check if RAG analysis mentions similar companies
        mentions_similar = any(company['company_name'].lower() in rag_analysis.lower() 
                              for company in similar_companies)
        print(f"✅ RAG analysis mentions similar companies: {mentions_similar}")
        
        print("\n🎉 RAG Implementation Test Results:")
        print("=" * 60)
        print(f"✅ Database Search: {'PASS' if similar_companies else 'FAIL'}")
        print(f"✅ Company Existence Check: {'PASS' if company_check else 'FAIL'}")
        print(f"✅ RAG Analysis Generation: {'PASS' if rag_analysis else 'FAIL'}")
        print(f"✅ Similar Company Detection: {'PASS' if edtech_companies else 'FAIL'}")
        print(f"✅ Knowledge Base Access: {'PASS' if beelinguapp_in_kb else 'FAIL'}")
        print(f"✅ Context Integration: {'PASS' if mentions_similar else 'FAIL'}")
        
        # Overall assessment
        all_tests_passed = all([
            similar_companies,
            company_check,
            rag_analysis,
            edtech_companies,
            beelinguapp_in_kb,
            mentions_similar
        ])
        
        print(f"\n🏆 Overall RAG Test Result: {'✅ ALL TESTS PASSED' if all_tests_passed else '❌ SOME TESTS FAILED'}")
        
        if all_tests_passed:
            print("\n🚀 RAG implementation is working perfectly!")
            print("The system can:")
            print("- Search the knowledge base for similar companies")
            print("- Retrieve relevant information about Beelinguapp and similar EdTech companies")
            print("- Generate contextual analysis using retrieved knowledge")
            print("- Integrate database insights into AI-generated content")
        
        return all_tests_passed
        
    except Exception as e:
        print(f"❌ RAG test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_specific_beelinguapp_retrieval():
    """Test specific retrieval of Beelinguapp data."""
    print("\n🔍 Testing Specific Beelinguapp Data Retrieval")
    print("=" * 50)
    
    try:
        rag_system = SimpleRAG()
        
        # Test direct Beelinguapp lookup
        beelinguapp_data = rag_system.knowledge_base.get('beelinguapp')
        
        if beelinguapp_data:
            print("✅ Beelinguapp data found in knowledge base:")
            print(json.dumps(beelinguapp_data, indent=2))
            
            # Verify key fields
            required_fields = ['company_name', 'industry', 'business_model', 'competitive_advantage']
            missing_fields = [field for field in required_fields if field not in beelinguapp_data]
            
            if not missing_fields:
                print("✅ All required fields present in Beelinguapp data")
                return True
            else:
                print(f"❌ Missing fields: {missing_fields}")
                return False
        else:
            print("❌ Beelinguapp data not found in knowledge base")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Beelinguapp retrieval: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting RAG Implementation Tests with Beelinguapp Data")
    print("=" * 70)
    
    # Test 1: General RAG functionality
    test1_result = test_rag_with_beelinguapp_data()
    
    # Test 2: Specific Beelinguapp retrieval
    test2_result = test_specific_beelinguapp_retrieval()
    
    print("\n" + "=" * 70)
    print("📊 FINAL TEST SUMMARY")
    print("=" * 70)
    print(f"General RAG Functionality: {'✅ PASS' if test1_result else '❌ FAIL'}")
    print(f"Beelinguapp Data Retrieval: {'✅ PASS' if test2_result else '❌ FAIL'}")
    
    overall_success = test1_result and test2_result
    print(f"\n🏆 Overall Result: {'✅ RAG IMPLEMENTATION VERIFIED' if overall_success else '❌ RAG NEEDS FIXES'}")
    
    if overall_success:
        print("\n🎉 RAG is working perfectly with Beelinguapp data!")
        print("The system successfully:")
        print("- Retrieves Beelinguapp information from the knowledge base")
        print("- Finds similar EdTech companies")
        print("- Generates contextual analysis using retrieved knowledge")
        print("- Integrates database insights into pitch deck generation")
    else:
        print("\n⚠️  RAG implementation needs attention.")
        print("Please check the error messages above for specific issues.")
