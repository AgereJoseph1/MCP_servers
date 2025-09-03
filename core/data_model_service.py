"""Model service for logical data generation."""

import json
import re
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from openai import Client
from openai.types.chat import ChatCompletion
from datetime import datetime, UTC
import functools

from core.config import settings
from core.prompts.main import MERGE_PROMPT, LOGICAL_MODEL_PROMPT, FINAL_REPLACEMENT_PROMPT, UPDATE_LOGICAL_MODEL_PROMPT
from core.exceptions import LLMServiceException, ValidationException
from core.logging_config import get_logger
from core.lms_service import retrieve_most_similar_table
from schemas.data_model import LogicalDataModel, LogicalPhysicalModel
from schemas.lms_service import LMSQueryRequest

logger = get_logger(__name__)

# Cache for repeated LLM calls (excluding message field for dynamic responses)
@functools.lru_cache(maxsize=100)
def cached_generate_logical_model(query: str) -> LogicalDataModel:
    """Cached version of generate_logical_model for repeated queries."""
    model = generate_logical_model(query)
    
    # Make the message field dynamic and conversational
    import random
    import time
    
    # Create a more dynamic message based on the query and entities
    entity_names = [entity.name for entity in model.entities]
    relationship_count = len(model.relationships)
    
    # Generate a personalized message with varied opening phrases
    message_templates = [
        f"Here's a comprehensive data model for your {query.lower()} requirements. The model includes {len(entity_names)} key entities: {', '.join(entity_names[:3])}{' and more' if len(entity_names) > 3 else ''}. With {relationship_count} relationships defined, this will help you effectively manage your data and support your business objectives.",
        f"Your data model is ready! Designed specifically for your {query.lower()} needs, I've identified {len(entity_names)} core entities including {', '.join(entity_names[:2])}{' and others' if len(entity_names) > 2 else ''}, connected through {relationship_count} relationships to ensure data integrity and support your operational requirements.",
        f"Perfect! I've built a data model tailored to your {query.lower()} requirements. The model captures {len(entity_names)} essential entities like {', '.join(entity_names[:2])}{' and more' if len(entity_names) > 2 else ''}, with {relationship_count} relationships that will help you maintain data consistency and achieve your business goals.",
        f"Great! Here's your {query.lower()} data model. It features {len(entity_names)} key entities: {', '.join(entity_names[:3])}{' and more' if len(entity_names) > 3 else ''}, with {relationship_count} relationships that will help you effectively manage your data and support your business objectives.",
        f"Excellent! Your {query.lower()} model is complete. I've designed it with {len(entity_names)} core entities including {', '.join(entity_names[:2])}{' and others' if len(entity_names) > 2 else ''}, connected through {relationship_count} relationships to ensure data integrity and support your operational requirements."
    ]
    
    # Add some randomness to make responses more conversational
    random.seed(int(time.time()) % 1000)  # Use time-based seed for variety
    model.message = random.choice(message_templates)
    
    return model

def clear_logical_model_cache():
    """Clear the logical model cache to force fresh responses."""
    cached_generate_logical_model.cache_clear()
    logger.info("Logical model cache cleared")

def extract_json_from_string(s: str) -> Any:
    """
    Extract JSON from a string, handling code blocks and plain JSON.
    
    Args:
        s: Input string that may contain JSON
        
    Returns:
        Parsed JSON object or original string if not JSON
    """
    try:
        # Try to extract JSON from a code block
        match = re.search(r"```(?:json)?\n(.*?)```", s, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            logger.debug(f"Extracting JSON from code block: {json_str[:100]}...")
            return json.loads(json_str)
        
        # Try to parse the whole string as JSON
        logger.debug(f"Attempting to parse entire string as JSON: {s[:100]}...")
        return json.loads(s)
        
    except json.JSONDecodeError as e:
        logger.debug(f"JSON parsing failed: {e}. Returning original string.")
        return s
    except Exception as e:
        logger.error(f"Unexpected error in JSON extraction: {e}")
        return s

def generate_logical_data(messages: List[Dict[str, Any]], query: str) -> LogicalDataModel:
    """
    Generate logical data using the LLM.
    Returns a LogicalDataModel object (parsed/validated).
    """
    start_time = time.time()
    logger.info(f"Generating logical data for query: {query[:100]}...")
    try:
        # Input validation
        if not messages:
            raise ValidationException("Messages list cannot be empty")
        if not query or not query.strip():
            raise ValidationException("Query cannot be empty")
        client = Client(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout
        )
        logger.debug(f"LLM request - Model: {settings.llm_model}, Max tokens: {settings.llm_max_tokens}")
        chat_completion = client.beta.chat.completions.parse(
            model=settings.llm_model,
            messages=messages,
            response_format=LogicalDataModel,
            temperature=0,
            seed=35
        )
        parsed = chat_completion.choices[0].message.parsed
        if not parsed:
            raise LLMServiceException("Empty response from LLM service")
        processing_time = time.time() - start_time
        logger.info(f"LLM response generated in {processing_time:.2f}s. Parsed LogicalDataModel.")
        return parsed
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"LLM service failed after {processing_time:.2f}s: {str(e)}")
        if isinstance(e, (LLMServiceException, ValidationException)):
            raise
        raise LLMServiceException(f"LLM service error: {str(e)}") from e
    


def add_timestamp_and_fix_role(msg):
    if msg.get("role") not in ("user", "assistant"):
        msg["role"] = "assistant"
    if "timestamp" not in msg:
        msg["timestamp"] = datetime.now(UTC).isoformat()
    return msg

def order_chat_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Order chat history chronologically with most recent first, but ensure assistant messages come before user messages in each conversation turn.
    Ensures all messages have valid role and timestamp.
    """
    try:
        if not history:
            logger.debug("Empty history provided, returning empty list")
            return []
        
        # Sort messages by timestamp (most recent first)
        sorted_history = sorted(history, key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Group messages into conversation turns (assistant + user pairs)
        conversation_turns = []
        i = 0
        
        while i < len(sorted_history):
            current_msg = sorted_history[i]
            
            # If current message is assistant, look for corresponding user message
            if current_msg.get("role") == "assistant":
                # Find the next user message that might be a response to this assistant
                user_msg = None
                j = i + 1
                while j < len(sorted_history) and sorted_history[j].get("role") != "user":
                    j += 1
                
                if j < len(sorted_history):
                    user_msg = sorted_history[j]
                    # Create turn with assistant first, then user
                    conversation_turns.append([current_msg, user_msg])
                    # Remove the user message from consideration
                    sorted_history.pop(j)
                else:
                    # No corresponding user message found
                    conversation_turns.append([current_msg])
            
            # If current message is user, look for corresponding assistant message
            elif current_msg.get("role") == "user":
                # Find the next assistant message that might be a response to this user
                assistant_msg = None
                j = i + 1
                while j < len(sorted_history) and sorted_history[j].get("role") != "assistant":
                    j += 1
                
                if j < len(sorted_history):
                    assistant_msg = sorted_history[j]
                    # Create turn with assistant first, then user
                    conversation_turns.append([assistant_msg, current_msg])
                    # Remove the assistant message from consideration
                    sorted_history.pop(j)
                else:
                    # No corresponding assistant message found
                    conversation_turns.append([current_msg])
            
            i += 1
        
        # Flatten the conversation turns
        ordered = [msg for turn in conversation_turns for msg in turn]
        
        logger.debug(f"Ordered {len(history)} messages into {len(ordered)} messages (chronological with assistant first)")
        # Apply timestamp and role fix to all messages
        return [add_timestamp_and_fix_role(msg) for msg in ordered]
    except Exception as e:
        logger.error(f"Error ordering chat history: {e}")
        # Return original history if ordering fails
        return history

def validate_message_structure(message: Dict[str, Any]) -> bool:
    """
    Validate that a message has the required structure.
    
    Args:
        message: Message dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ["role", "content"]
    
    if not isinstance(message, dict):
        logger.warning("Message is not a dictionary")
        return False
    
    for field in required_fields:
        if field not in message:
            logger.warning(f"Message missing required field: {field}")
            return False
    
    if message["role"] not in ["user", "assistant"]:
        logger.warning(f"Invalid message role: {message['role']}")
        return False
    
    return True 

def merge_data_models(scores: dict, logical_model, physical_model) -> LogicalPhysicalModel:  
    """
    Merge logical and physical data models using LLM.
    Optimized for parallel processing and better performance.
    """
    start_time = time.time()
    logger.info("Starting optimized merge of logical and physical models")
    
    try:
        client = Client(
            base_url=settings.llm_base_url, 
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout
        )
         
        # Prepare the prompt with optimized context
        messages = [ 
            {"role": "system", "content": MERGE_PROMPT},
            {"role": "user", "content": f"Scores: {scores}\n\nLogical Model: {logical_model}\n\nPhysical Model: {physical_model}"}
        ]
        
        logger.debug(f"LLM request - Model: {settings.llm_model}, Max tokens: {settings.llm_max_tokens}")
        
        # Use structured output for better performance
        chat_completion = client.beta.chat.completions.parse(
            model=settings.llm_model,
            messages=messages,
            response_format=LogicalPhysicalModel,
            temperature=0,
            seed=35
        )
        
        parsed = chat_completion.choices[0].message.parsed
        if not parsed:
            raise LLMServiceException("Empty response from LLM service")
        
        processing_time = time.time() - start_time
        logger.info(f"Model merge completed in {processing_time:.2f}s")
        return parsed
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Model merge failed after {processing_time:.2f}s: {str(e)}")
        if isinstance(e, LLMServiceException):
            raise
        raise LLMServiceException(f"Model merge error: {str(e)}") from e

def generate_logical_model(query: str) -> LogicalDataModel:
    """
    Generate logical data using the LLM.
    Returns a LogicalDataModel object (parsed/validated).
    """
    start_time = time.time()
    logger.info(f"Generating logical data for query: {query[:100]}...")
    messages = [{"role": "system", "content": LOGICAL_MODEL_PROMPT}, {"role": "user", "content": query}]
    try:
        # Input validation
        if not messages:
            raise ValidationException("Messages list cannot be empty")
        if not query or not query.strip():
            raise ValidationException("Query cannot be empty")
        client = Client(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout
        )
        logger.debug(f"LLM request - Model: {settings.llm_model}, Max tokens: {settings.llm_max_tokens}")
        chat_completion = client.beta.chat.completions.parse(
            model=settings.llm_model,
            messages=messages,
            response_format=LogicalDataModel,
            temperature=0,
            seed=35
        )
        parsed = chat_completion.choices[0].message.parsed
        if not parsed:
            raise LLMServiceException("Empty response from LLM service")
        processing_time = time.time() - start_time
        logger.info(f"LLM response generated in {processing_time:.2f}s. Parsed LogicalDataModel.")
        return parsed
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"LLM service failed after {processing_time:.2f}s: {str(e)}")
        if isinstance(e, (LLMServiceException, ValidationException)):
            raise
        raise LLMServiceException(f"LLM service error: {str(e)}") from e
    
def gen_q(query: str):
    messages = [{"role": "system", "content": "Based on a user query, generate a one or two word phrase that we can use to categorise the query so that we can search an index for that domain."}, {"role": "user", "content": query}]
    client = Client(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout
        )
        
    chat_completion = client.beta.chat.completions.parse(
        model=settings.llm_model,
        messages=messages,
        temperature=0,
        seed=35
    )

    print(chat_completion)

    response = chat_completion.choices[0].message.content
    return response

async def calculate_table_scores(entities, env_name=None):
    """Calculate table scores for all entities in parallel for better performance."""
    logger.info(f"Processing {len(entities)} entities in parallel")
    
    # Create tasks for all entities to process them in parallel
    tasks = []
    for entity in entities:
        request_query = LMSQueryRequest(
            model_name=env_name,  
            entity=entity,
            limit=5
        )
        task = retrieve_most_similar_table(request_query)
        tasks.append(task)

    # Execute all tasks in parallel
    scores = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out any exceptions and log them
    valid_scores = []
    for i, score in enumerate(scores):
        if isinstance(score, Exception):
            logger.error(f"Error processing entity {i}: {score}")
        else:
            valid_scores.append(score)
    
    logger.info(f"Successfully processed {len(valid_scores)} entities")
    return valid_scores


def final_merge(scores, merged_model):
    """
    Final merge step with optimized performance and better error handling.
    """
    start_time = time.time()
    logger.info("Starting final merge optimization")
    
    try:
        client = Client(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout,
        )

        messages = [{"role": "system", "content": FINAL_REPLACEMENT_PROMPT}]
        user_content = {"role": "user", "content": f"Scores: {scores} \n\n Merged Data Model: {merged_model}"}
        messages.append(user_content)

        logger.debug(f"Final merge LLM request - Model: {settings.llm_model}")
        
        response = client.beta.chat.completions.parse(
            model=settings.llm_model,
            messages=messages,
            response_format=LogicalPhysicalModel,
            temperature=0,
            seed=35
        ) 
        
        parsed = response.choices[0].message.parsed
        if not parsed:
            raise LLMServiceException("Empty response from LLM service")
        
        processing_time = time.time() - start_time
        logger.info(f"Final merge completed in {processing_time:.2f}s")
        return parsed
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Final merge failed after {processing_time:.2f}s: {str(e)}")
        raise LLMServiceException(f"Final merge error: {str(e)}") from e


def llm_update_logical_model(current_model: Dict[str, Any], instruction: str) -> LogicalPhysicalModel:
    """Use the LLM to apply a minimal update to an existing LogicalPhysicalModel.

    - Preserves structure and relationships unless explicitly instructed.
    - Returns a validated LogicalPhysicalModel.
    """
    try:
        client = Client(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout,
        )
        messages = [
            {"role": "system", "content": UPDATE_LOGICAL_MODEL_PROMPT},
            {"role": "user", "content": f"Instruction: {instruction}\n\nCurrentModel: {json.dumps(current_model)}"},
        ]
        response = client.beta.chat.completions.parse(
            model=settings.llm_model,
            messages=messages,
            response_format=LogicalPhysicalModel,
            temperature=0,
            seed=35,
        )
        parsed = response.choices[0].message.parsed
        if not parsed:
            raise LLMServiceException("Empty response from LLM for model update")
        return parsed
    except Exception as e:
        logger.error(f"LLM update failed: {e}")
        raise
