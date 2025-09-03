import math
from typing import List, Tuple, Dict, Any
import functools
import asyncio
from core.config import settings
from core.logging_config import get_logger
from fastapi import HTTPException
from httpx import AsyncClient, HTTPStatusError, ReadTimeout, Limits
from schemas.lms_service import LMSQueryDocument, LMSQueryRequest

logger = get_logger(__name__)

# Optimized HTTP client with connection pooling
_http_client = None

async def get_http_client() -> AsyncClient:
    """Get or create an optimized HTTP client with connection pooling."""
    global _http_client
    if _http_client is None:
        _http_client = AsyncClient(
            timeout=settings.llm_timeout,
            limits=Limits(max_connections=100, max_keepalive_connections=20),
            http2=True
        )
    return _http_client

async def close_http_client():
    """Close the HTTP client when shutting down."""
    global _http_client
    if _http_client:
        await _http_client.aclose()
        _http_client = None

# Cache for embeddings to avoid repeated API calls
@functools.lru_cache(maxsize=1000)
def cached_embedding_key(texts: Tuple[str, ...]) -> str:
    """Create a cache key for embedding requests."""
    return "|".join(sorted(texts))

# Global cache for embeddings
_embedding_cache = {}

def cosine_similarity_simple(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors using basic math."""
    if len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)

def calculate_similarity_matrix_simple(embeddings1: List[List[float]], embeddings2: List[List[float]]) -> List[List[float]]:
    """Calculate similarity matrix between two sets of embeddings."""
    matrix = []
    for emb1 in embeddings1:
        row = []
        for emb2 in embeddings2:
            similarity = cosine_similarity_simple(emb1, emb2)
            row.append(similarity)
        matrix.append(row)
    return matrix

async def get_embedding_batch(texts: List[str], batch_size: int = 50) -> List[List[float]]:
    """
    Get embeddings for a list of texts in batches for better performance.
    """
    if not texts:
        return []
    
    # Check cache first
    cache_key = cached_embedding_key(tuple(sorted(texts)))
    if cache_key in _embedding_cache:
        logger.debug(f"Using cached embeddings for {len(texts)} texts")
        return _embedding_cache[cache_key]
    
    logger.info(f"Getting embeddings for {len(texts)} texts in batches of {batch_size}")
    
    all_embeddings = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            batch_embeddings = await get_embedding(batch)
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error(f"Error getting embeddings for batch {i//batch_size}: {e}")
            # Return empty embeddings for failed batch
            all_embeddings.extend([[0.0] * 1536] * len(batch))  # Assuming 1536 dimensions
    
    # Cache the result
    _embedding_cache[cache_key] = all_embeddings
    
    return all_embeddings


async def get_tables_from_lms(table_name: str, pre_filter: dict, limit: int = 10) -> List[Tuple[LMSQueryDocument, float]]:
    try:
        logger.info(settings.lms_vector_store)
        client = await get_http_client()
        body = {"query": table_name, "k": limit, "pre_filter": pre_filter}
        response = await client.post(url=settings.lms_vector_store, json=body)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Successfully fetched {table_name} from LMS")
        return data
    except HTTPStatusError as e:
        logger.error(f"HTTP error while getting embedding: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail="Embedding service error")
    except ReadTimeout:
        logger.error("Timeout while getting embedding")
        raise HTTPException(status_code=504, detail="Embedding service timeout")
    except Exception as e:
        logger.error(f"Unexpected error while getting embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def get_logical_models_from_kb(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Query the QuestLogicalModels knowledge base for existing logical models.
    
    Args:
        query: Search query for logical models
        limit: Maximum number of results to return
        
    Returns:
        List of logical model documents with similarity scores
    """
    try: 
        logger.info(f"Querying QuestLogicalModels knowledge base for: {query}")
        client = await get_http_client()
        body = {"query": query, "k": limit, "pre_filter": {"index_name": "QuestLogicalModels"}}
        response = await client.post(url=settings.lms_logical_models_store, json=body)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Successfully fetched {len(data)} logical models from knowledge base")
        logger.debug(f"Knowledge base response: {data}")
        return data
    except HTTPStatusError as e:
        logger.error(f"HTTP error while querying knowledge base: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail="Knowledge base service error")
    except ReadTimeout:
        logger.error("Timeout while querying knowledge base")
        raise HTTPException(status_code=504, detail="Knowledge base service timeout")
    except Exception as e:
        logger.error(f"Unexpected error while querying knowledge base: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def search_similar_logical_models(entity_names: List[str], model_context: str = "", limit: int = 3) -> List[Dict[str, Any]]:
    """
    Search for similar logical models in the knowledge base based on entity names and context.
    
    Args:
        entity_names: List of entity names to search for
        model_context: Additional context about the model being created
        limit: Maximum number of similar models to return
        
    Returns:
        List of similar logical models with metadata
    """
    try:
        # Create a search query combining entity names and context
        search_terms = " ".join(entity_names)
        if model_context:
            search_terms += f" {model_context}"
        
        logger.info(f"Searching for similar logical models with terms: {search_terms}")
        
        # Query the knowledge base
        results = await get_logical_models_from_kb(search_terms, limit=limit)
        
        # Filter and rank results based on relevance
        relevant_models = []
        for result in results:
            if isinstance(result, tuple) and len(result) >= 2:
                document, score = result
                if score >= 0.3:  # Minimum similarity threshold
                    relevant_models.append({
                        "document": document,
                        "similarity_score": score,
                        "model_name": document.get("page_content", "Unknown Model"),
                        "metadata": document.get("metadata", {}).get("metadata", {})
                    })
        
        # Sort by similarity score (highest first)
        relevant_models.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        logger.info(f"Found {len(relevant_models)} relevant logical models")
        return relevant_models
        
    except Exception as e:
        logger.error(f"Error searching for similar logical models: {e}")
        return []


async def get_embedding(texts: List[str]) -> List[List[float]]:
    """
    Get the embedding for a list of text values using the LLM embedding service.
    """
    try:
        client = await get_http_client()
        response = await client.post(url=settings.lms_embedding_url, json=texts)
        response.raise_for_status()
        data = response.json()
        return data["embeddings"]
    except HTTPStatusError as e:
        logger.error(f"HTTP error while getting embedding: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail="Embedding service error")
    except ReadTimeout:
        logger.error("Timeout while getting embedding")
        raise HTTPException(status_code=504, detail="Embedding service timeout")
    except Exception as e:
        logger.error(f"Unexpected error while getting embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error while getting embedding: {e}")


async def calculate_similarity(column_embedding: List[float], lms_columns_embedding: List[List[float]]) -> float:
    """
    Calculate the maximum cosine similarity between a column embedding and a list of LMS embeddings.
    """
    try:
        similarities = []
        for stored_vector in lms_columns_embedding:
            similarity = cosine_similarity_simple(column_embedding, stored_vector)
            similarities.append(similarity)
        
        max_similarity = max(similarities) if similarities else 0.0
        return float(max_similarity)
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        raise HTTPException(status_code=500, detail="Similarity calculation error")


async def assign_column_similarities(
        query_result: List[Tuple[LMSQueryDocument, float]],
        all_column_embeddings: List[List[float]],
        entity_columns_embeddings: List[List[float]],
) -> None:
    """
    Assigns a `match` score for each column based on maximum similarity with any entity column embedding.
    """
    try:
        # Calculate similarity matrix using simple method
        similarity_matrix = calculate_similarity_matrix_simple(all_column_embeddings, entity_columns_embeddings)
        
        # Find max similarities for each column
        max_similarities = []
        for row in similarity_matrix:
            max_similarities.append(max(row) if row else 0.0)

        idx = 0
        for document, _ in query_result:
            for column in document['metadata']['columns']:
                if idx >= len(max_similarities):
                    logger.warning("Column embedding index out of bounds")
                    break
                column['match'] = float(max_similarities[idx])
                idx += 1
    except Exception as e:
        logger.error(f"Error assigning column similarities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def assign_environment_similarities(
        query_result: List[Tuple[LMSQueryDocument, float]],
        all_environment_embeddings: List[List[float]],
        model_name_embeddings: List[float],
) -> None:
    """
    Assigns an `environmentMatch` score for each document based on similarity to the model name embedding.
    """
    try:
        # Calculate similarities between environment embeddings and model name embedding
        similarities = []
        for env_embedding in all_environment_embeddings:
            similarity = cosine_similarity_simple(env_embedding, model_name_embeddings)
            similarities.append(similarity)

        for idx, (document, _) in enumerate(query_result):
            if idx >= len(similarities):
                logger.warning("Environment embedding index out of bounds")
                break
            document['metadata']['metadata']['environmentMatch'] = float(similarities[idx])
    except Exception as e:
        logger.error(f"Error assigning environment similarities: {e}")
        raise HTTPException(status_code=500, detail="Environment similarity assignment error")


async def adjust_score(query_result: List[Tuple[LMSQueryDocument, float]], w_base: float, w_columns: float, w_auth: float ,entity_col_length: int) -> List[Tuple[LMSQueryDocument, float]]:
    """
    Adjusts the LMS scores using a weighted formula that includes:
    - The original score
    - Average of column similarity scores (only those >= 0.75)
    - Environment similarity score
    """
    try:
        adjusted_documents = []

        for document, score in query_result:
            columns = document['metadata']['columns']
            column_scores = [column.get('match') for column in columns if column.get('match') is not None and column.get('match') >= 0.75]
            if len(column_scores) >= entity_col_length:
                column_scores.sort(reverse=True)
                column_scores = column_scores[:entity_col_length]
            adjusted_col_score = sum(column_scores) / entity_col_length
            authorized = document['metadata'].get('authorize', 0.0)

            adjusted_score = (w_base * score) + (w_columns * adjusted_col_score) + (w_auth * authorized)
            adjusted_documents.append((document, adjusted_score))

        return adjusted_documents
    except Exception as e:
        logger.error(f"Error adjusting scores: {e}")
        raise HTTPException(status_code=500, detail="Score adjustment error: " + str(e))

# Assuming LMSQueryDocument, LMSQueryRequest, get_embedding, 
# get_tables_from_lms, assign_column_similarities, adjust_score, and logger are already defined

async def get_table_with_max_score(query_result: List[Tuple[LMSQueryDocument, float]]):
    try:
        if not query_result:
            return []

        document, _ = max(query_result, key=lambda item: item[1])
        return document

    except Exception as e:
        logger.error(f"Error selecting table with max score: {e}")
        raise HTTPException(status_code=500, detail="Max score selection error: " + str(e))


async def retrieve_most_similar_table(request: LMSQueryRequest) -> List[Tuple[LMSQueryDocument, float]]:
    try:
        model_name = request.model_name
        entity = request.entity
        limit = request.limit
        w_base = request.w_base
        w_columns = request.w_columns
        w_auth = request.w_auth

        logger.info(f"Querying LMS service for entity: {entity.name}")
        
        pre_filter = {}
        if model_name:
            pre_filter = {'environmentName': model_name}

        print(f"Querying LMS service for entity: {entity.name}")

        # Step 1: Get entity column embeddings (now handled in parallel below)
        entity_columns = [attr.name for attr in entity.attributes]

        # Step 2: Fetch candidate tables from LMS
        query_result: List[Tuple[LMSQueryDocument, float]] = await get_tables_from_lms(
            entity.name, limit=limit, pre_filter=pre_filter
        )

        logger.info(f"{len(query_result)} tables returned from LMS service")

        if len(query_result) == 0:
            logger.info(f"No tables returned from LMS service with name: {entity.name} from logical model: {model_name}")
            return []

        # Collect all column names
        all_column_names = []
        for document, _ in query_result:
            all_column_names.extend([col['name'] for col in document['metadata']['columns']])

        logger.info(f"Embedding {len(all_column_names)} column names")

        # Get embeddings in parallel with entity embeddings
        embedding_tasks = [
            get_embedding_batch(all_column_names),
            get_embedding_batch(entity_columns)
        ]
        
        all_column_embeddings, entity_columns_embeddings = await asyncio.gather(*embedding_tasks)

        # Assign similarity scores
        await assign_column_similarities(query_result, all_column_embeddings, entity_columns_embeddings)

        # Step 6: Adjust document scores
        tables_with_adjusted_weights = await adjust_score(
            query_result, w_base, w_columns, w_auth, len(entity_columns)
        )

        # Return table with the highest match
        # return await get_table_with_max_score(tables_with_adjusted_weights)

        
        table_name_and_scores_dict = {}
        max_score = 0
        best_match = {}
        for document, score in tables_with_adjusted_weights:
            table_name_and_scores_dict[document['metadata']['tableName']] = score
            best_match[document['metadata']['tableName']] = [column['name'] for column in document['metadata']['columns']]
            

    
        result = {entity.name: table_name_and_scores_dict, "best_match": best_match }
        logger.info(result)
        return result

    except HTTPException as e:
        logger.error(f"HTTPException during LMS query: {e}")
        raise e

    except Exception as e:
        logger.error(f"Unexpected error during LMS query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_latest_model_from_session_history(session_id: str) -> Dict[str, Any]:
    """
    Get the latest logical model from conversation history for a given session.
    
    Args:
        session_id: The session ID to get the model from
        
    Returns:
        The latest logical model from the conversation history, or None if not found
    """
    try:
        from core.storage import get_chat_history
        import uuid
        
        # Convert string session_id to UUID if needed
        if isinstance(session_id, str):
            try:
                session_uuid = uuid.UUID(session_id)
            except ValueError:
                # If it's not a valid UUID, use it as is (for simple session IDs like "1")
                session_uuid = session_id
        else:
            session_uuid = session_id
            
        # Get chat history for this session
        chat_history = await get_chat_history(session_uuid)
        
        if not chat_history:
            logger.warning(f"No chat history found for session {session_id}")
            return None
            
        # Find the latest assistant message that contains a model
        for message in reversed(chat_history):  # Start from most recent
            if message.get("role") == "assistant":
                content = message.get("content")
                if content and isinstance(content, dict):
                    # Check if this looks like a logical model
                    if "entities" in content and "relationships" in content:
                        logger.info(f"Found latest model in session {session_id}")
                        return content
                        
        logger.warning(f"No logical model found in chat history for session {session_id}")
        return None
        
    except Exception as e:
        logger.error(f"Error getting latest model from session history: {e}")
        return None
