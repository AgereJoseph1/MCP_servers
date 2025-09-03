#!/usr/bin/env python3
"""
Example usage of the new vector store query MCP tool endpoint.

This script demonstrates how to use the /api/v1/query-vector-store endpoint
to search the FIBORAG vector store for relevant information.
"""

import asyncio
import httpx
import json
from typing import Dict, Any

async def query_vector_store(query: str, k: int = 4, pre_filter: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Query the FIBORAG vector store using the new MCP tool endpoint.
    
    Args:
        query: The search query string
        k: Number of results to return (default: 4)
        pre_filter: Optional pre-filter criteria (default: {})
    
    Returns:
        Dictionary containing the query results
    """
    if pre_filter is None:
        pre_filter = {}
    
    # Local API endpoint
    url = "http://localhost:8000/api/v1/reference-fibo"
    
    payload = {
        "query": query,
        "k": k,
        "pre_filter": pre_filter
    }
    
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"Error querying vector store: {e}")
        return None

async def main():
    """Example usage of the vector store query endpoint."""
    
    print("🔍 Vector Store Query Examples")
    print("=" * 50)
    
    # Example 1: Query about Bank for International Settlements
    print("\n1. Querying about Bank for International Settlements...")
    result1 = await query_vector_store(
        query="BankForInternational Settlements (bank for international settlements)",
        k=4
    )
    
    if result1:
        print(f"✅ Query: {result1['query']}")
        print(f"📊 Total results: {result1['total_results']}")
        print(f"📝 Results: {len(result1['results'])} items")
        
        # Show first result preview
        if result1['results']:
            first_result = result1['results'][0]
            print(f"📄 First result preview: {str(first_result)[:200]}...")
    
    # Example 2: Query about financial regulations
    print("\n2. Querying about financial regulations...")
    result2 = await query_vector_store(
        query="financial regulations compliance requirements",
        k=3
    )
    
    if result2:
        print(f"✅ Query: {result2['query']}")
        print(f"📊 Total results: {result2['total_results']}")
        print(f"📝 Results: {len(result2['results'])} items")
    
    # Example 3: Query with pre-filter
    print("\n3. Querying with pre-filter...")
    result3 = await query_vector_store(
        query="risk management",
        k=2,
        pre_filter={"category": "finance"}  # Example pre-filter
    )
    
    if result3:
        print(f"✅ Query: {result3['query']}")
        print(f"📊 Total results: {result3['total_results']}")
        print(f"📝 Results: {len(result3['results'])} items")

if __name__ == "__main__":
    print("🚀 Starting vector store query examples...")
    print("⚠️  Make sure the server is running on localhost:8000")
    print()
    
    asyncio.run(main())
    
    print("\n✨ Examples completed!")
    print("\n💡 You can now use this endpoint in your MCP tools or LLM orchestration!") 