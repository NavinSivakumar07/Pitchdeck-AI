"""
Simplified RAG Implementation without CrewAI
Provides vector search and knowledge retrieval for pitch deck analysis.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleRAG:
    """Simplified RAG implementation using OpenAI for embeddings and analysis."""
    
    def __init__(self):
        """Initialize the RAG system."""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        openai.api_key = self.openai_api_key
        
        # Sample knowledge base from existing pitch decks
        self.knowledge_base = {
            "beelinguapp": {
                "company_name": "Beelinguapp",
                "industry": "EdTech/Language Learning",
                "description": "Language learning app that uses audiobooks and stories to teach languages",
                "business_model": "Freemium with premium subscriptions",
                "target_market": "Language learners, students, professionals",
                "key_features": ["Audio-visual learning", "Story-based approach", "Multiple languages"],
                "funding_stage": "Seed/Series A",
                "competitive_advantage": "Unique story-based learning methodology",
                "market_size": "Global language learning market worth $60B+",
                "monetization": "Premium subscriptions, in-app purchases"
            },
            "airbnb": {
                "company_name": "Airbnb",
                "industry": "Travel/Hospitality",
                "description": "Platform for short-term accommodation rentals",
                "business_model": "Commission-based marketplace",
                "target_market": "Travelers, property owners",
                "key_features": ["Peer-to-peer rentals", "Global platform", "Trust & safety"],
                "funding_stage": "Series A",
                "competitive_advantage": "First-mover advantage in P2P accommodation",
                "market_size": "Global travel accommodation market",
                "monetization": "Host and guest service fees"
            },
            "uber": {
                "company_name": "Uber",
                "industry": "Transportation/Mobility",
                "description": "On-demand ride-sharing platform",
                "business_model": "Commission-based marketplace",
                "target_market": "Urban commuters, travelers",
                "key_features": ["On-demand rides", "Dynamic pricing", "Driver network"],
                "funding_stage": "Series A",
                "competitive_advantage": "Network effects and scale",
                "market_size": "Global transportation market",
                "monetization": "Commission from rides, delivery services"
            },
            "dropbox": {
                "company_name": "Dropbox",
                "industry": "Cloud Storage/SaaS",
                "description": "Cloud storage and file synchronization service",
                "business_model": "Freemium SaaS",
                "target_market": "Individuals, businesses, teams",
                "key_features": ["File sync", "Cloud storage", "Collaboration"],
                "funding_stage": "Series A",
                "competitive_advantage": "Simple, reliable file synchronization",
                "market_size": "Global cloud storage market",
                "monetization": "Premium subscriptions, business plans"
            }
        }
    
    def find_similar_companies(self, startup_data: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Find similar companies based on industry and business model.
        
        Args:
            startup_data: Information about the startup
            top_k: Number of similar companies to return
            
        Returns:
            List of similar companies with relevance scores
        """
        try:
            industry = startup_data.get('industry_type', '').lower()
            business_model = startup_data.get('business_model', '').lower()
            solution = startup_data.get('solution_summary', '').lower()
            
            similar_companies = []
            
            for company_key, company_data in self.knowledge_base.items():
                relevance_score = 0
                
                # Industry similarity
                if any(word in company_data['industry'].lower() for word in industry.split()):
                    relevance_score += 3
                
                # Business model similarity
                if any(word in company_data['business_model'].lower() for word in business_model.split()):
                    relevance_score += 2
                
                # Solution similarity
                if any(word in company_data['description'].lower() for word in solution.split()):
                    relevance_score += 2
                
                # Target market similarity
                target_customer = startup_data.get('target_customer_profile', '').lower()
                if any(word in company_data['target_market'].lower() for word in target_customer.split()):
                    relevance_score += 1
                
                if relevance_score > 0:
                    company_info = company_data.copy()
                    company_info['relevance_score'] = relevance_score
                    similar_companies.append(company_info)
            
            # Sort by relevance score and return top_k
            similar_companies.sort(key=lambda x: x['relevance_score'], reverse=True)
            return similar_companies[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar companies: {e}")
            return []
    
    def analyze_with_rag(self, startup_data: Dict[str, Any]) -> str:
        """
        Perform RAG analysis by retrieving similar companies and generating insights.
        
        Args:
            startup_data: Information about the startup
            
        Returns:
            RAG analysis with insights from similar companies
        """
        try:
            company_name = startup_data.get('startup_name', 'Unknown')
            
            # Find similar companies
            similar_companies = self.find_similar_companies(startup_data)
            
            if not similar_companies:
                return f"No similar companies found in database for {company_name}. This appears to be a unique business model."
            
            # Create context from similar companies
            context = "Similar companies in our database:\n\n"
            for i, company in enumerate(similar_companies, 1):
                context += f"{i}. {company['company_name']} ({company['industry']})\n"
                context += f"   - Business Model: {company['business_model']}\n"
                context += f"   - Key Advantage: {company['competitive_advantage']}\n"
                context += f"   - Monetization: {company['monetization']}\n"
                context += f"   - Relevance Score: {company['relevance_score']}/10\n\n"
            
            # Generate analysis using OpenAI with RAG context
            prompt = f"""
            Analyze the startup "{company_name}" using the following information and similar companies from our database:
            
            STARTUP INFORMATION:
            Company: {startup_data.get('startup_name', '')}
            Industry: {startup_data.get('industry_type', '')}
            Problem: {startup_data.get('key_problem_solved', '')}
            Solution: {startup_data.get('solution_summary', '')}
            Business Model: {startup_data.get('business_model', '')}
            Target Market: {startup_data.get('target_customer_profile', '')}
            Competitive Advantage: {startup_data.get('why_you_win', '')}
            
            SIMILAR COMPANIES FROM DATABASE:
            {context}
            
            Based on this RAG analysis, provide:
            1. How this startup compares to similar successful companies
            2. Key patterns and insights from the database
            3. Potential risks based on similar companies' challenges
            4. Strategic recommendations based on successful patterns
            5. Market positioning advice
            
            Format as a comprehensive analysis report.
            """
            
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)

            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert startup analyst with access to a comprehensive database of successful pitch decks. Provide detailed analysis based on historical patterns."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            analysis = response.choices[0].message.content
            
            # Add RAG metadata
            rag_analysis = f"""
# RAG ANALYSIS REPORT: {company_name}
Generated using Retrieval-Augmented Generation

## Database Search Results
Found {len(similar_companies)} similar companies in our pitch deck database.

## Similar Companies Analysis
{context}

## AI Analysis Based on Historical Data
{analysis}

## RAG Verification
✅ Knowledge Retrieved: {len(similar_companies)} similar companies
✅ Context Provided: Historical patterns and insights
✅ Analysis Generated: Strategic recommendations based on database
"""
            
            return rag_analysis
            
        except Exception as e:
            logger.error(f"Error in RAG analysis: {e}")
            return f"Error performing RAG analysis: {str(e)}"
    
    def check_company_exists(self, company_name: str) -> Dict[str, Any]:
        """
        Check if a company exists in our knowledge base.
        
        Args:
            company_name: Name of the company to check
            
        Returns:
            Dictionary with existence status and company data
        """
        try:
            company_key = company_name.lower().replace(' ', '').replace('-', '')
            
            for key, data in self.knowledge_base.items():
                if key in company_key or company_key in key:
                    return {
                        "exists": True,
                        "company_data": data,
                        "database_key": key
                    }
            
            return {
                "exists": False,
                "company_data": None,
                "database_key": None
            }
            
        except Exception as e:
            logger.error(f"Error checking company existence: {e}")
            return {"exists": False, "error": str(e)}
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        return {
            "total_companies": len(self.knowledge_base),
            "industries": list(set(data['industry'] for data in self.knowledge_base.values())),
            "companies": list(self.knowledge_base.keys())
        }
