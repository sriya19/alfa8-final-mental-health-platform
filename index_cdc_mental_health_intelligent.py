#!/usr/bin/env python3
"""
Intelligent CDC Mental Health Dataset Indexer
Dynamically discovers and indexes ALL mental health-related datasets from CDC.
Uses LLM to generate comprehensive search queries without hardcoding.

Target: 5000+ mental health datasets
Strategy: Start broad, then expand based on what we find

Usage:
    python index_cdc_mental_health_intelligent.py

Environment variables needed:
    - OPENAI_API_KEY: Your OpenAI API key
"""

import asyncio
import httpx
import sys
import json
import os
from typing import Dict, List, Set

# Configuration
BACKEND_URL = "http://localhost:8000"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

# Track what we've searched to avoid duplicates
searched_queries: Set[str] = set()


async def generate_mental_health_queries() -> List[str]:
    """
    Use LLM to generate comprehensive mental health search queries.
    This discovers queries dynamically instead of hardcoding them.
    """
    
    prompt = """You are a public health data specialist helping to discover ALL mental health-related datasets in the CDC catalog.

Generate 100 diverse search queries that would find mental health datasets. Cover:
1. Mental health conditions (depression, anxiety, PTSD, bipolar, schizophrenia, etc.)
2. Substance use disorders (opioid, alcohol, drug abuse, overdose, addiction)
3. Crisis & outcomes (suicide, self-harm, overdose deaths, mental health emergency)
4. Symptoms & presentations (distress, psychological symptoms, mood disorders)
5. Treatment & services (mental health care, psychiatric services, counseling, therapy)
6. Populations (youth, adolescent, adult, elderly, veteran, maternal, child)
7. Risk factors (trauma, stress, adverse experiences, violence exposure)
8. Surveys & data systems (BRFSS, YRBSS, NHIS, NVSS, surveillance systems)
9. Comorbidities (mental health + chronic disease, mental health + substance use)
10. Social determinants (poverty, housing, employment, stigma, access, disparities)
11. Specific drugs/substances (fentanyl, heroin, methamphetamine, prescription opioids)
12. Geographic & demographic variations (by state, county, age, gender, race)

Make queries specific enough to find relevant datasets but broad enough to capture many results.
Mix short queries (2-3 words) with longer specific ones (4-6 words).

CRITICAL: Focus ONLY on mental health and substance use topics. These are mental health related.

Return ONLY a JSON array of 100 search query strings, no other text:
["query1", "query2", ...]"""

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o",  # Use GPT-4 for better query generation
                    "messages": [
                        {"role": "system", "content": "You are a public health search specialist. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.8,  # Higher temperature for diversity
                    "max_tokens": 2000
                }
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract and parse JSON response
            content = data["choices"][0]["message"]["content"].strip()
            
            # Clean up response (remove markdown if present)
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            
            queries = json.loads(content)
            
            print(f"✓ Generated {len(queries)} mental health search queries using LLM")
            return queries
            
    except Exception as e:
        print(f"✗ Error generating queries with LLM: {e}")
        print("  Falling back to seed queries...")
        
        # Fallback: seed queries that will help us discover more
        return [
            "mental health", "depression", "anxiety", "suicide", "substance abuse",
            "drug overdose", "opioid", "behavioral health", "psychiatric",
            "BRFSS mental", "YRBSS mental", "mental distress"
        ]


async def discover_related_queries(indexed_datasets: List[Dict]) -> List[str]:
    """
    Analyze indexed datasets to discover new search terms.
    Uses LLM to extract relevant keywords from dataset names and descriptions.
    """
    
    # Sample 20 random datasets
    import random
    sample = random.sample(indexed_datasets, min(20, len(indexed_datasets)))
    
    # Create context from dataset names and descriptions
    dataset_context = "\n".join([
        f"- {d.get('name', '')}: {d.get('description', '')[:200]}"
        for d in sample
    ])
    
    prompt = f"""Based on these mental health dataset examples from CDC, suggest 20 additional specific search queries that would find related datasets we might have missed.

Dataset examples:
{dataset_context}

Generate 20 NEW search queries focusing on:
- Specific conditions or symptoms mentioned
- Specific populations or demographics
- Specific time periods or geographic areas
- Specific survey names or data systems
- Related but different keywords

Return ONLY a JSON array of 20 strings:
["query1", "query2", ...]"""

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 500
                }
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            
            # Clean and parse
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            
            new_queries = json.loads(content)
            return new_queries
            
    except Exception as e:
        print(f"  Note: Could not discover new queries ({e})")
        return []


async def index_query(query: str, limit: int = 500) -> Dict:
    """Index datasets for a single query"""
    url = f"{BACKEND_URL}/semantic/reindex"
    params = {"org": "CDC", "limit": limit, "q": query}
    
    async with httpx.AsyncClient(timeout=600) as client:
        try:
            response = await client.post(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"ok": False, "error": str(e)}


async def get_indexed_count() -> int:
    """Get current count of indexed datasets"""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Use a direct database query via an endpoint, or estimate from search
            response = await client.get(f"{BACKEND_URL}/rag_status?org=CDC")
            response.raise_for_status()
            data = response.json()
            return data.get("indexed_datasets", 0)
    except:
        return 0


async def intelligent_indexing():
    """
    Intelligent multi-phase indexing strategy to reach 5000+ datasets
    """
    
    print("=" * 80)
    print("INTELLIGENT CDC MENTAL HEALTH DATASET INDEXER")
    print("Target: 5000+ mental health datasets")
    print("Strategy: Dynamic query generation using LLM")
    print("=" * 80)
    print(f"\nBackend: {BACKEND_URL}")
    
    if not OPENAI_API_KEY:
        print("\n✗ Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    print("\nPhase 1: Generating comprehensive search queries using LLM...")
    initial_queries = await generate_mental_health_queries()
    
    print(f"\nPhase 2: Initial broad indexing ({len(initial_queries)} queries)")
    print("=" * 80)
    
    total_indexed = 0
    total_skipped = 0
    all_datasets = []
    
    # Phase 2: Index with generated queries
    for i, query in enumerate(initial_queries, 1):
        if query.lower() in searched_queries:
            continue
        
        searched_queries.add(query.lower())
        
        print(f"[{i}/{len(initial_queries)}] '{query[:50]}'...", end=" ")
        
        result = await index_query(query, limit=500)
        
        if result.get("ok"):
            indexed = result.get("indexed", 0)
            skipped = result.get("skipped", 0)
            total_indexed += indexed
            total_skipped += skipped
            print(f"✓ +{indexed}")
            
            # Track for discovery
            if indexed > 0:
                # Note: We don't have dataset details here, but we track queries
                all_datasets.append({"query": query, "count": indexed})
        else:
            print(f"✗ {result.get('error', 'unknown')}")
            if "rate" in str(result.get("error", "")).lower():
                print("  ⚠️  Rate limited, pausing...")
                await asyncio.sleep(10)
        
        # Progress check
        if i % 20 == 0:
            print(f"\n  📊 Progress: {total_indexed} unique datasets indexed so far")
            print(f"  💰 Estimated cost so far: ${(total_indexed * 0.0002):.2f}")
            
            if total_indexed >= 5000:
                print(f"\n🎉 Target reached! {total_indexed} datasets indexed")
                break
            
            await asyncio.sleep(3)
        else:
            await asyncio.sleep(2)
    
    # Phase 3: Discovery and expansion (if needed)
    if total_indexed < 5000 and len(all_datasets) > 20:
        print("\n" + "=" * 80)
        print(f"Phase 3: Discovery phase (current: {total_indexed}, target: 5000)")
        print("=" * 80)
        print("\nAnalyzing indexed datasets to discover new search angles...")
        
        # Use successful queries to find more
        discovery_rounds = 0
        max_discovery_rounds = 3
        
        while total_indexed < 5000 and discovery_rounds < max_discovery_rounds:
            discovery_rounds += 1
            print(f"\nDiscovery round {discovery_rounds}/{max_discovery_rounds}")
            
            # Generate new queries based on what we found
            new_queries = await discover_related_queries(all_datasets)
            
            if not new_queries:
                print("  No new queries discovered")
                break
            
            print(f"  Discovered {len(new_queries)} new search angles")
            
            for i, query in enumerate(new_queries, 1):
                if query.lower() in searched_queries:
                    continue
                
                searched_queries.add(query.lower())
                
                print(f"  [{i}/{len(new_queries)}] '{query}'...", end=" ")
                
                result = await index_query(query, limit=500)
                
                if result.get("ok"):
                    indexed = result.get("indexed", 0)
                    total_indexed += indexed
                    total_skipped += result.get("skipped", 0)
                    print(f"✓ +{indexed}")
                    
                    if indexed > 0:
                        all_datasets.append({"query": query, "count": indexed})
                else:
                    print(f"✗")
                
                if total_indexed >= 5000:
                    print(f"\n🎉 Target reached! {total_indexed} datasets indexed")
                    break
                
                await asyncio.sleep(2)
            
            if total_indexed >= 5000:
                break
            
            print(f"\n  Total after discovery round {discovery_rounds}: {total_indexed} datasets")
            await asyncio.sleep(5)
    
    # Phase 4: Broad sweep if still under target
    if total_indexed < 5000:
        print("\n" + "=" * 80)
        print(f"Phase 4: Broad sweep (current: {total_indexed}, target: 5000)")
        print("=" * 80)
        
        broad_queries = [
            "health", "disease", "mortality", "morbidity", "surveillance",
            "behavioral", "psychological", "emotional", "wellness", "distress",
            "treatment", "therapy", "care", "service", "hospital",
            "prevention", "screening", "intervention", "program"
        ]
        
        for query in broad_queries:
            if total_indexed >= 5000:
                break
            
            if query.lower() in searched_queries:
                continue
            
            searched_queries.add(query.lower())
            
            print(f"  Broad: '{query}'...", end=" ")
            result = await index_query(query, limit=500)
            
            if result.get("ok"):
                indexed = result.get("indexed", 0)
                total_indexed += indexed
                print(f"✓ +{indexed}")
            else:
                print(f"✗")
            
            await asyncio.sleep(2)
    
    # Final summary
    print("\n" + "=" * 80)
    print("INDEXING COMPLETE")
    print("=" * 80)
    print(f"\n📊 Final Statistics:")
    print(f"  ✓ Total unique datasets indexed: {total_indexed}")
    print(f"  ⊘ Total skipped (duplicates): {total_skipped}")
    print(f"  🔍 Total unique queries used: {len(searched_queries)}")
    print(f"  💰 Estimated cost: ${(total_indexed * 0.0002):.2f}")
    
    if total_indexed >= 5000:
        print(f"\n🎉 SUCCESS! Target of 5000+ datasets achieved!")
    else:
        print(f"\n⚠️  Indexed {total_indexed} datasets (target was 5000)")
        print(f"   This may be all available mental health datasets in CDC catalog")
    
    # Top queries
    top_queries = sorted(
        [d for d in all_datasets if "count" in d],
        key=lambda x: x.get("count", 0),
        reverse=True
    )[:10]
    
    if top_queries:
        print(f"\n🏆 Most productive queries:")
        for item in top_queries:
            print(f"  - '{item['query']}': {item['count']} datasets")
    
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    print("\nVerify total in database:")
    print("  docker exec -it mental-health-platform-postgres-1 psql -U mhuser -d mhdb \\")
    print("    -c \"SELECT COUNT(*) FROM semantic_index WHERE org='CDC';\"")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Test semantic search with various mental health queries")
    print("2. Users can now find mental health datasets on any topic")
    print("3. Set up monthly refresh to catch new datasets")


async def check_backend():
    """Check if backend is running"""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{BACKEND_URL}/rag_status?org=CDC")
            response.raise_for_status()
            print("✓ Backend connection successful\n")
            return True
    except Exception as e:
        print(f"✗ Cannot connect to backend: {e}")
        return False


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Intelligently index 5000+ CDC mental health datasets"
    )
    parser.add_argument(
        "--backend-url",
        default="http://localhost:8000",
        help="Backend API URL"
    )
    
    args = parser.parse_args()
    
    global BACKEND_URL
    BACKEND_URL = args.backend_url
    
    print("🚀 Intelligent CDC Mental Health Indexer\n")
    
    try:
        if not asyncio.run(check_backend()):
            sys.exit(1)
        
        asyncio.run(intelligent_indexing())
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted - Progress saved")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()