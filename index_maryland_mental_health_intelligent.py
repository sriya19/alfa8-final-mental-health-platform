#!/usr/bin/env python3
"""
Intelligent LLM-Powered Maryland Mental Health Data Indexer
Uses OpenAI GPT-4 to dynamically generate and refine search strategies
Target: 10,000+ Maryland/Baltimore mental health records
"""

import asyncio
import httpx
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Tuple
import logging
import random
from collections import defaultdict

import pandas as pd
from sodapy import Socrata
from Bio import Entrez
import psycopg2
from psycopg2.extras import Json, execute_batch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o"  
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
SOCRATA_APP_TOKEN = os.getenv("SOCRATA_APP_TOKEN", "")

# Database configuration
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "database": os.getenv("POSTGRES_DB", "mh_catalog"),
    "user": os.getenv("POSTGRES_USER", "app_user"),
    "password": os.getenv("POSTGRES_PASSWORD", "app_user"),
    "port": int(os.getenv("POSTGRES_PORT", 5432))
}

# Tracking
searched_queries: Set[str] = set()
indexed_hashes: Set[str] = set()
successful_patterns: List[str] = []  # Track what works
dataset_samples: List[Dict] = []  # Sample data for LLM analysis

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================
# DATABASE FUNCTIONS
# ============================================

def get_db_connection():
    """Create database connection"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        sys.exit(1)


def save_to_database(records: List[Dict]) -> tuple:
    """Save records to database with deduplication"""
    if not records:
        return 0, 0
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    inserted = 0
    skipped = 0
    
    try:
        for record in records:
            content_hash = hashlib.sha256(
                f"{record['source']}_{record.get('title', '')}_{record['content']}".encode()
            ).hexdigest()
            
            if content_hash in indexed_hashes:
                skipped += 1
                continue
            
            try:
                cur.execute("""
                    INSERT INTO maryland_mental_health_data 
                    (source, dataset_id, query_used, location, title, content, metadata, url, data_hash)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (data_hash) DO NOTHING
                    RETURNING id
                """, (
                    record['source'],
                    record.get('dataset_id'),
                    record.get('query_used'),
                    record.get('location'),
                    record.get('title'),
                    record['content'],
                    Json(record.get('metadata', {})),
                    record.get('url'),
                    content_hash
                ))
                
                if cur.fetchone():
                    inserted += 1
                    indexed_hashes.add(content_hash)
                    # Track successful patterns
                    if record.get('query_used') and inserted == 1:
                        successful_patterns.append(record['query_used'])
                else:
                    skipped += 1
                    
            except Exception as e:
                logger.debug(f"Insert error: {e}")
                skipped += 1
        
        conn.commit()
        
    except Exception as e:
        logger.error(f"Batch save error: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()
    
    return inserted, skipped


# ============================================
# LLM-POWERED QUERY GENERATION
# ============================================

async def generate_initial_queries_llm() -> List[str]:
    """Use GPT-4 to generate comprehensive initial queries"""
    
    if not OPENAI_API_KEY:
        logger.error("No OpenAI API key!")
        return []
    
    prompt = """You are a public health data specialist focusing on Maryland and Baltimore mental health datasets.

Generate 150 diverse, specific search queries that would find mental health datasets for Maryland/Baltimore.

Consider ALL of these aspects:
1. Maryland counties (all 24): Allegany, Anne Arundel, Baltimore, Calvert, Caroline, Carroll, Cecil, Charles, Dorchester, Frederick, Garrett, Harford, Howard, Kent, Montgomery, Prince George's, Queen Anne's, Somerset, St. Mary's, Talbot, Washington, Wicomico, Worcester
2. Major cities: Baltimore, Columbia, Germantown, Silver Spring, Waldorf, Glen Burnie, Frederick, Rockville, Gaithersburg, Bethesda, Towson, Bowie, Annapolis
3. ZIP codes: 206xx-219xx (Maryland range), especially 212xx (Baltimore)
4. Institutions: Johns Hopkins, University of Maryland, Sheppard Pratt, Spring Grove Hospital, MedStar, Adventist HealthCare
5. Conditions: depression, anxiety, PTSD, bipolar, schizophrenia, OCD, ADHD, autism, eating disorders, personality disorders
6. Substances: opioid, heroin, fentanyl, alcohol, cocaine, methamphetamine, prescription drugs, marijuana
7. Demographics: youth, adolescent, teen, adult, elderly, veteran, LGBTQ, minority, immigrant, homeless
8. Services: crisis intervention, emergency department, inpatient, outpatient, telehealth, peer support, recovery
9. Programs: PBHS, Maryland Medicaid, Baltimore BCRI, Maryland BHA initiatives
10. Time periods: 2015-2024, quarterly data, monthly reports
11. Data types: surveillance, statistics, facilities, providers, insurance claims, mortality, hospitalizations

Create queries that combine multiple aspects (e.g., "Baltimore youth depression 2023", "Prince George's County opioid crisis emergency").

Return ONLY a JSON array of 150 unique query strings. Mix short (2-3 words) and long (4-7 words) queries.
Focus on queries that would return actual datasets, not general information."""

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": OPENAI_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a Maryland public health data expert. Return only valid JSON arrays."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.8,
                    "max_tokens": 3000
                }
            )
            response.raise_for_status()
            data = response.json()
            # --- Token usage / cost logger ---
            usage = (data or {}).get("usage") or {}
            pt = usage.get("prompt_tokens", 0)
            ct = usage.get("completion_tokens", 0)
            tt = usage.get("total_tokens", pt + ct)

            MODEL_PRICES = {
                "gpt-4o": {"input": 0.000005, "output": 0.000015},
                "gpt-4o-mini": {"input": 0.000003, "output": 0.000006},
            }

            m = OPENAI_MODEL
            inp_cost = MODEL_PRICES.get(m, {}).get("input", 0.0)
            out_cost = MODEL_PRICES.get(m, {}).get("output", 0.0)
            est_cost = tt * max(inp_cost, out_cost)

            logger.info(
                f"[OpenAI usage] model={m} "
                f"prompt_tokens={pt} completion_tokens={ct} "
                f"total_tokens={tt} est_cost=${est_cost:.6f}"
            )

            content = data["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.split("```")[1].replace("json", "").strip()
            
            queries = json.loads(content)
            logger.info(f"✅ LLM generated {len(queries)} initial queries")
            return queries
            
    except Exception as e:
        logger.error(f"LLM query generation failed: {e}")
        return []


async def discover_queries_from_data(sample_data: List[Dict], iteration: int) -> List[str]:
    """Analyze fetched data to discover new search patterns"""
    
    if not OPENAI_API_KEY or not sample_data:
        return []
    
    # Prepare context from successful data
    context_items = []
    for item in sample_data[:30]:  # Sample 30 items
        context_items.append({
            'source': item.get('source'),
            'title': item.get('title', '')[:100],
            'location': item.get('location'),
            'keywords': extract_keywords(item.get('content', ''))
        })
    
    prompt = f"""Analyze these successfully indexed Maryland mental health records and generate NEW search queries.

Iteration: {iteration}
Successfully indexed samples:
{json.dumps(context_items, indent=2)[:3000]}

Previous successful query patterns:
{json.dumps(successful_patterns[-20:], indent=2)}

Based on patterns you see in the data, generate 50 NEW specific queries that would find SIMILAR but DIFFERENT datasets.

Focus on:
1. Specific locations mentioned but not fully explored
2. Time periods or years referenced
3. Specific programs or initiatives mentioned
4. Demographics or populations referenced
5. New combinations of successful terms
6. Variations of dataset names you see
7. Related conditions or comorbidities
8. Specific Maryland/Baltimore neighborhoods or regions
9. Healthcare systems or providers mentioned
10. Grant numbers, project codes, or dataset IDs referenced

BE CREATIVE and SPECIFIC. Don't repeat queries already used.

Return ONLY a JSON array of 50 new query strings."""

    try:
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={
                    "model": OPENAI_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are discovering new search patterns from data. Return only JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.9,  # Higher creativity
                    "max_tokens": 1500
                }
            )
            data = response.json()
            # --- Token usage / cost logger ---
            usage = (data or {}).get("usage") or {}
            pt = usage.get("prompt_tokens", 0)
            ct = usage.get("completion_tokens", 0)
            tt = usage.get("total_tokens", pt + ct)

            MODEL_PRICES = {
                "gpt-4o": {"input": 0.000005, "output": 0.000015},
                "gpt-4o-mini": {"input": 0.000003, "output": 0.000006},
            }

            m = OPENAI_MODEL
            inp_cost = MODEL_PRICES.get(m, {}).get("input", 0.0)
            out_cost = MODEL_PRICES.get(m, {}).get("output", 0.0)
            est_cost = tt * max(inp_cost, out_cost)

            logger.info(
                f"[OpenAI usage] model={m} "
                f"prompt_tokens={pt} completion_tokens={ct} "
                f"total_tokens={tt} est_cost=${est_cost:.6f}"
            )

            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"].strip()
                if content.startswith("```"):
                    content = content.split("```")[1].replace("json", "").strip()
                
                queries = json.loads(content)
                logger.info(f"✅ LLM discovered {len(queries)} new queries from data patterns")
                return queries
    except Exception as e:
        logger.error(f"Discovery failed: {e}")
    
    return []


async def generate_targeted_queries(target_source: str, current_count: int) -> List[str]:
    """Generate queries specifically targeted at a source that needs more data"""
    
    if not OPENAI_API_KEY:
        return []
    
    source_context = {
        'CDC': "CDC WONDER, BRFSS, YRBSS, NVSS mortality data, WISQARS, surveillance systems",
        'PubMed': "Johns Hopkins, University of Maryland, NIH research, clinical trials, Maryland medical journals",
        'Maryland Open Data': "Maryland state datasets, county statistics, SHIP data, MDH reports, local initiatives",
        'SAMHSA': "Treatment facilities, substance abuse services, behavioral health providers, TEDS data"
    }
    
    prompt = f"""Generate 30 highly specific queries to find more {target_source} data about Maryland/Baltimore mental health.

Current {target_source} records: {current_count} (need more!)

{target_source} context: {source_context.get(target_source, '')}

Generate queries that would specifically work well for {target_source}:
- Use terminology and formats that {target_source} uses
- Include Maryland-specific filters that work for this source
- Target datasets we haven't found yet

Return ONLY a JSON array of 30 query strings optimized for {target_source}."""

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={
                    "model": "gpt-4o-mini",  # Faster model for targeted queries
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 800
                }
            )
            data = response.json()
            # --- Token usage / cost logger ---
            usage = (data or {}).get("usage") or {}
            pt = usage.get("prompt_tokens", 0)
            ct = usage.get("completion_tokens", 0)
            tt = usage.get("total_tokens", pt + ct)

            MODEL_PRICES = {
                "gpt-4o": {"input": 0.000005, "output": 0.000015},
                "gpt-4o-mini": {"input": 0.000003, "output": 0.000006},
            }

            m = OPENAI_MODEL
            inp_cost = MODEL_PRICES.get(m, {}).get("input", 0.0)
            out_cost = MODEL_PRICES.get(m, {}).get("output", 0.0)
            est_cost = tt * max(inp_cost, out_cost)

            logger.info(
                f"[OpenAI usage] model={m} "
                f"prompt_tokens={pt} completion_tokens={ct} "
                f"total_tokens={tt} est_cost=${est_cost:.6f}"
            )

            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"].strip()
                if content.startswith("```"):
                    content = content.split("```")[1].replace("json", "").strip()
                return json.loads(content)
    except Exception as e:
        logger.error(f"Targeted query generation failed: {e}")
    
    return []


def extract_keywords(text: str) -> List[str]:
    """Extract keywords from text for LLM context"""
    import re
    # Simple keyword extraction
    words = re.findall(r'\b[A-Za-z]{4,}\b', text.lower())
    # Filter common words
    stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'their', 'more'}
    keywords = [w for w in words if w not in stopwords]
    return list(set(keywords[:10]))  # Top 10 unique keywords


# ============================================
# ENHANCED DATA FETCHERS
# ============================================

async def fetch_cdc_comprehensive(query: str, limit: int = 500) -> List[Dict]:
    """Fetch comprehensive CDC data"""
    records = []
    
    try:
        client = Socrata("data.cdc.gov", SOCRATA_APP_TOKEN)
        
        # Search for datasets
        datasets = client.datasets(q=f"{query}", limit=20)
        
        for dataset in datasets:
            if 'resource' in dataset and 'id' in dataset['resource']:
                dataset_id = dataset['resource']['id']
                
                # Multiple filter strategies for Maryland
                filters = [
                    "state='MD'",
                    "state='Maryland'", 
                    "stateabbr='MD'",
                    "locationabbr='MD'",
                    "locationdesc LIKE '%Maryland%'",
                    "locationdesc LIKE '%Baltimore%'",
                    "geographic_location LIKE '%Maryland%'"
                ]
                
                for filter_clause in filters:
                    try:
                        results = client.get(dataset_id, where=filter_clause, limit=min(limit, 200))
                        
                        for item in results:
                            if is_maryland_related(str(item)):
                                records.append({
                                    'source': 'CDC',
                                    'dataset_id': dataset_id,
                                    'query_used': query,
                                    'location': extract_location(item),
                                    'title': f"{dataset.get('resource', {}).get('name', '')} - {query}",
                                    'content': json.dumps(item),
                                    'metadata': {
                                        'dataset_name': dataset.get('resource', {}).get('name'),
                                        'attribution': dataset.get('resource', {}).get('attribution')
                                    },
                                    'url': f"https://data.cdc.gov/d/{dataset_id}"
                                })
                        
                        if results:
                            break  # Found data with this filter
                            
                    except Exception as e:
                        continue  # Try next filter
        
        client.close()
        
    except Exception as e:
        logger.error(f"CDC fetch error: {e}")
    
    return records


async def fetch_maryland_opendata_all(query: str = "", limit: int = 2000) -> List[Dict]:
    """Fetch ALL Maryland Open Data mental health records"""
    records = []
    
    # Comprehensive list of Maryland mental health datasets
    all_datasets = {
        # Mental Health
        "xiqr-nb66": "Emergency Department Visits Related to Mental Health",
        "r6y2-6v7r": "Behavioral Health Grantees",
        "spv3-bsy5": "Psychiatric Hospitals",
        "q7qa-mgpa": "Mental Health Services",
        "t8wg-z9cr": "Substance Abuse Treatment",
        
        # Substance Abuse
        "e72v-x8kf": "Drug and Alcohol Deaths",
        "mgd3-qk8t": "Crisis Services",
        "unw6-2nwn": "Mental Health Providers",
        "vsw6-hfrd": "Substance Abuse Providers",
        "f4jx-q2gm": "Behavioral Health Administration",
        
        # Healthcare
        "xy7w-s4wm": "Hospital Utilization",
        "62yf-j4qe": "Health Professional Shortage Areas",
        "nssq-numg": "Primary Care Access",
        
        # Social Services
        "y2cb-5b5v": "Social Services Programs",
        "bqyi-jqvg": "Community Health Resources",
        "rm6m-z79a": "Prevention Programs",
        
        # Demographics
        "ryxx-aeaf": "Health Disparities",
        "p73e-yfqr": "Population Health Metrics"
    }
    
    try:
        client = Socrata("opendata.maryland.gov", SOCRATA_APP_TOKEN)
        
        # Fetch from all known datasets
        for dataset_id, description in all_datasets.items():
            try:
                offset = 0
                dataset_records = []
                
                while offset < limit:
                    # Get records in batches
                    if query:
                        # Search within dataset
                        results = client.get(
                            dataset_id,
                            q=query,
                            limit=500,
                            offset=offset
                        )
                    else:
                        # Get all records
                        results = client.get(
                            dataset_id,
                            limit=500,
                            offset=offset
                        )
                    
                    if not results:
                        break
                    
                    for item in results:
                        records.append({
                            'source': 'Maryland Open Data',
                            'dataset_id': dataset_id,
                            'query_used': query or f"Full dataset: {description}",
                            'location': extract_location(item),
                            'title': f"{description} - Record {offset + results.index(item)}",
                            'content': json.dumps(item),
                            'metadata': {
                                'dataset_name': description,
                                'portal': 'opendata.maryland.gov',
                                'record_number': offset + results.index(item)
                            },
                            'url': f"https://opendata.maryland.gov/d/{dataset_id}"
                        })
                    
                    offset += len(results)
                    
                    # Rate limiting
                    await asyncio.sleep(0.3)
                
                if dataset_records:
                    logger.info(f"  Maryland dataset {dataset_id}: {len(dataset_records)} records")
                
            except Exception as e:
                logger.debug(f"Error with dataset {dataset_id}: {e}")
                continue
        
        # Also search for additional datasets
        if query:
            try:
                new_datasets = client.datasets(q=query, limit=15)
                
                for dataset in new_datasets:
                    if 'resource' in dataset and 'id' in dataset['resource']:
                        dataset_id = dataset['resource']['id']
                        
                        if dataset_id not in all_datasets:
                            try:
                                results = client.get(dataset_id, limit=100)
                                
                                for item in results:
                                    records.append({
                                        'source': 'Maryland Open Data',
                                        'dataset_id': dataset_id,
                                        'query_used': query,
                                        'location': extract_location(item),
                                        'title': dataset.get('resource', {}).get('name', ''),
                                        'content': json.dumps(item),
                                        'metadata': dataset.get('resource', {}),
                                        'url': f"https://opendata.maryland.gov/d/{dataset_id}"
                                    })
                            except:
                                pass
            except:
                pass
        
        client.close()
        
    except Exception as e:
        logger.error(f"Maryland Open Data error: {e}")
    
    return records


async def fetch_pubmed_comprehensive(query: str, max_results: int = 200) -> List[Dict]:
    """Fetch comprehensive PubMed data"""
    if not NCBI_API_KEY:
        return []
    
    records = []
    Entrez.email = "research@example.com"
    Entrez.api_key = NCBI_API_KEY
    
    # Build comprehensive search query
    maryland_terms = [
        "Maryland[Affiliation]",
        "Baltimore[Affiliation]",
        "Johns Hopkins[Affiliation]",
        "University of Maryland[Affiliation]",
        "Sheppard Pratt[Affiliation]",
        "NIH Clinical Center[Affiliation]",
        "Walter Reed[Affiliation]",
        "Maryland[Title]",
        "Baltimore[Title]"
    ]
    
    # Combine with query
    search_query = f"({query}) AND ({' OR '.join(maryland_terms)})"
    
    try:
        # Search
        handle = Entrez.esearch(
            db="pubmed",
            term=search_query,
            retmax=max_results,
            sort="relevance",
            retmode="json"
        )
        search_results = json.loads(handle.read())
        handle.close()
        
        # Safely extract PMIDs - handle various response formats
        pmids = []
        if isinstance(search_results, dict):
            esearchresult = search_results.get("esearchresult", {})
            if isinstance(esearchresult, dict):
                idlist = esearchresult.get("idlist", [])
                # Ensure idlist is actually a list
                if isinstance(idlist, list):
                    pmids = idlist
                elif isinstance(idlist, (str, int)):
                    pmids = [str(idlist)]
                else:
                    logger.debug(f"Unexpected idlist type: {type(idlist)}")
                    pmids = []
        
        if not pmids:
            logger.debug(f"No PMIDs found for query: {query}")
            return records
        
        # Convert all PMIDs to strings
        pmids = [str(pmid) for pmid in pmids]
        
        # Fetch details in batches
        for i in range(0, len(pmids), 50):
            batch_pmids = pmids[i:i+50]
            
            try:
                handle = Entrez.efetch(
                    db="pubmed",
                    id=",".join(batch_pmids),
                    rettype="abstract",
                    retmode="text"
                )
                
                content = handle.read()
                handle.close()
                
                # Parse abstracts
                abstracts = content.split("\n\n")
                
                for abstract in abstracts:
                    if abstract.strip() and len(abstract) > 100:
                        # Generate unique hash for ID
                        abstract_hash = str(abs(hash(abstract)))[:8]
                        
                        records.append({
                            'source': 'PubMed',
                            'dataset_id': f"PMID_{abstract_hash}",
                            'query_used': query,
                            'location': 'Maryland',
                            'title': extract_pubmed_title(abstract),
                            'content': abstract,
                            'metadata': {
                                'search_query': search_query,
                                'database': 'PubMed'
                            },
                            'url': 'https://pubmed.ncbi.nlm.nih.gov/'
                        })
                
                await asyncio.sleep(0.3)  # NCBI rate limit
                
            except Exception as e:
                logger.error(f"Error fetching PubMed batch: {e}")
                continue
                
    except Exception as e:
        logger.error(f"PubMed search error: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    
    return records


# ============================================
# HELPER FUNCTIONS
# ============================================

def extract_location(item: Dict) -> str:
    """Extract location from data item"""
    location_fields = [
        'location', 'city', 'county', 'jurisdiction', 
        'geographic_area', 'locationdesc', 'county_name'
    ]
    for field in location_fields:
        if field in item and item[field]:
            loc = str(item[field])
            if 'maryland' in loc.lower() or 'baltimore' in loc.lower() or 'md' in loc.lower():
                return loc
    return 'Maryland'


def is_maryland_related(text: str) -> bool:
    """Check if text is Maryland-related"""
    text_lower = text.lower()
    # Expanded Maryland indicators
    indicators = [
        'maryland', 'baltimore', ' md ', 'md.', 'balt', 'annapolis',
        'johns hopkins', 'university of maryland', 'sheppard pratt'
    ]
    return any(indicator in text_lower for indicator in indicators)


def extract_pubmed_title(abstract: str) -> str:
    """Extract title from PubMed abstract"""
    lines = abstract.split('\n')
    for line in lines[:5]:  # Check first 5 lines
        if line.strip() and not line.startswith(' ') and len(line) > 20:
            return line.strip()[:200]
    return f"PubMed Article {str(hash(abstract))[:8]}"


# ============================================
# MAIN LLM-POWERED ORCHESTRATION
# ============================================

async def intelligent_llm_indexing():
    """Main indexing using LLM intelligence"""
    
    print("=" * 80)
    print("🤖 LLM-POWERED MARYLAND MENTAL HEALTH INDEXER")
    print(f"Model: {OPENAI_MODEL}")
    print("Target: 10,000+ records")
    print("=" * 80)
    
    if not OPENAI_API_KEY:
        print("❌ OpenAI API key required!")
        return
    
    total_inserted = 0
    total_skipped = 0
    iteration = 0
    
    # Phase 1: LLM generates initial queries
    print("\n📚 Phase 1: LLM generating initial queries...")
    initial_queries = await generate_initial_queries_llm()
    
    if not initial_queries:
        print("❌ Failed to generate queries")
        return
    
    print(f"✅ Generated {len(initial_queries)} initial queries")
    
    # Phase 2: Fetch all Maryland Open Data first
    print("\n📊 Phase 2: Fetching ALL Maryland Open Data...")
    maryland_data = await fetch_maryland_opendata_all(limit=3000)
    
    if maryland_data:
        inserted, skipped = save_to_database(maryland_data)
        total_inserted += inserted
        total_skipped += skipped
        dataset_samples.extend(maryland_data[:50])  # Save samples for LLM
        print(f"  Maryland Open Data: {inserted} new, {skipped} duplicates")
    
    # Phase 3: Use initial queries
    print(f"\n🔍 Phase 3: Processing {len(initial_queries)} LLM queries...")
    
    for i, query in enumerate(initial_queries, 1):
        if total_inserted >= 10000:
            break
        
        if query.lower() in searched_queries:
            continue
        
        searched_queries.add(query.lower())
        
        if i % 10 == 0:
            print(f"\n[{i}/{len(initial_queries)}] Progress: {total_inserted} records")
        
        # Fetch from all sources in parallel
        tasks = [
            fetch_cdc_comprehensive(query, limit=300),
            fetch_pubmed_comprehensive(query, max_results=100),
            fetch_maryland_opendata_all(query, limit=200)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_records = []
        for result in results:
            if not isinstance(result, Exception):
                all_records.extend(result)
        
        if all_records:
            inserted, skipped = save_to_database(all_records)
            total_inserted += inserted
            total_skipped += skipped
            
            # Save successful samples
            if inserted > 0:
                dataset_samples.extend(all_records[:5])
        
        # Rate limiting
        await asyncio.sleep(0.5)
    
    # Phase 4: LLM discovers patterns and generates new queries
    while total_inserted < 10000 and iteration < 5:
        iteration += 1
        
        print(f"\n🧠 Phase 4.{iteration}: LLM analyzing patterns...")
        print(f"Current total: {total_inserted} records")
        
        # Get source statistics
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT source, COUNT(*) FROM maryland_mental_health_data GROUP BY source")
        source_stats = dict(cur.fetchall())
        cur.close()
        conn.close()
        
        print(f"Source distribution: {source_stats}")
        
        # Generate targeted queries for underrepresented sources
        for source, count in source_stats.items():
            if count < 2000:  # Need more from this source
                print(f"\n🎯 Generating targeted queries for {source}...")
                targeted = await generate_targeted_queries(source, count)
                
                for query in targeted[:10]:  # Process top 10
                    if query.lower() not in searched_queries:
                        searched_queries.add(query.lower())
                        
                        if source == 'CDC':
                            records = await fetch_cdc_comprehensive(query, 200)
                        elif source == 'PubMed':
                            records = await fetch_pubmed_comprehensive(query, 100)
                        else:
                            records = await fetch_maryland_opendata_all(query, 200)
                        
                        if records:
                            inserted, skipped = save_to_database(records)
                            total_inserted += inserted
                            total_skipped += skipped
                        
                        if total_inserted >= 10000:
                            break
        
        # Discover new patterns from successful data
        if dataset_samples:
            print(f"\n🔮 LLM discovering new patterns from {len(dataset_samples)} samples...")
            discovered_queries = await discover_queries_from_data(
                random.sample(dataset_samples, min(50, len(dataset_samples))),
                iteration
            )
            
            print(f"✅ Discovered {len(discovered_queries)} new queries")
            
            for query in discovered_queries:
                if total_inserted >= 10000:
                    break
                
                if query.lower() not in searched_queries:
                    searched_queries.add(query.lower())
                    
                    # Fetch from all sources
                    tasks = [
                        fetch_cdc_comprehensive(query, 150),
                        fetch_pubmed_comprehensive(query, 75),
                        fetch_maryland_opendata_all(query, 150)
                    ]
                    
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    all_records = []
                    for result in results:
                        if not isinstance(result, Exception):
                            all_records.extend(result)
                    
                    if all_records:
                        inserted, skipped = save_to_database(all_records)
                        total_inserted += inserted
                        total_skipped += skipped
                        dataset_samples.extend(all_records[:3])
                    
                    await asyncio.sleep(0.5)
    
    # Final Summary
    print("\n" + "=" * 80)
    print("🎉 INDEXING COMPLETE")
    print("=" * 80)
    print(f"\n📊 Final Statistics:")
    print(f"  ✓ Total records inserted: {total_inserted:,}")
    print(f"  ⊘ Total duplicates skipped: {total_skipped:,}")
    print(f"  🔍 Total unique queries used: {len(searched_queries):,}")
    print(f"  🧠 LLM iterations: {iteration}")
    
    # Get final statistics
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("SELECT COUNT(*) FROM maryland_mental_health_data")
    total = cur.fetchone()[0]
    
    cur.execute("SELECT source, COUNT(*) FROM maryland_mental_health_data GROUP BY source")
    source_stats = cur.fetchall()
    
    print(f"\n📈 Total records in database: {total:,}")
    print("\n📊 Records by source:")
    for source, count in source_stats:
        print(f"  - {source}: {count:,}")
    
    cur.execute("SELECT COUNT(DISTINCT location) FROM maryland_mental_health_data")
    locations = cur.fetchone()[0]
    print(f"\n📍 Unique locations: {locations:,}")
    
    cur.close()
    conn.close()
    
    if total_inserted >= 10000:
        print("\n🏆 SUCCESS! Reached 10,000+ records!")
    
    # Estimated cost
    api_calls = len(searched_queries) + iteration * 2  # Rough estimate
    estimated_cost = api_calls * 0.01  # Rough cost estimate
    print(f"\n💰 Estimated OpenAI API cost: ${estimated_cost:.2f}")


async def main():
    """Main entry point"""
    try:
        await intelligent_llm_indexing()
    except KeyboardInterrupt:
        print("\n\nInterrupted - Progress saved to database")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())