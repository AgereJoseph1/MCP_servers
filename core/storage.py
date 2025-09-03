"""Storage and utility functions for the application."""

import time
from typing import List, Dict, Any
from datetime import datetime, timezone
from core.logging_config import get_logger
from core.exceptions import StorageException
import uuid
from motor.motor_asyncio import AsyncIOMotorClient
from core.config import settings

MONGO_URL = settings.mongo_url

mongo_client = AsyncIOMotorClient(MONGO_URL)

db = mongo_client['ADPC_db']


async def get_db():
    yield db

logger = get_logger(__name__)

# In-memory chat histories per user (no login, session_id required in header)
chat_histories: Dict[uuid.UUID, List[Dict[uuid.UUID, Any]]] = {}


def get_utc_timestamp() -> str:
    """
    Get current UTC timestamp in ISO format.

    Returns:
        ISO format UTC timestamp string
    """
    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        return timestamp
    except Exception as e:
        logger.error(f"Error generating UTC timestamp: {e}")
        # Fallback to simple timestamp
        return str(int(time.time()))

async def get_chat_history(session_id: uuid.UUID) -> List[Dict[str, Any]]:
    """
    Get chat history for a specific user.

    Args:
        session_id: User identifier

    Returns:
        List of chat messages for the user
    """
    try:
        logger.info(f"All chat histories: {chat_histories}")
        if not session_id:
            logger.warning("Empty session_id provided to get_chat_history")
            return []

        history = chat_histories.get(session_id, [])
        db_history = await db['conversation'].find_one({"_id": str(session_id)}, {'_id': 0})

        logger.debug(f"Retrieved {len(db_history['messages'])} messages for user {session_id}")
        print("from the db: ", db_history['messages'])
        return db_history['messages']

    except Exception as e:
        logger.error(f"Error retrieving chat history for user {session_id}: {e}")
        return []

def add_timestamp_and_fix_role(msg):
    # Only allow 'user' or 'assistant'
    if msg.get("role") not in ("user", "assistant"):
        msg["role"] = "assistant"
    # Add timestamp if missing
    if "timestamp" not in msg:
        msg["timestamp"] = datetime.utcnow().isoformat()
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

def is_valid_uuid(val):
    try:
        uuid.UUID(str(val))
        return True
    except Exception:
        return False


def clean_logical_model(model):
    if not isinstance(model, dict):
        return model
    entity_name_to_id = {}
    # Build a mapping from entity names to IDs
    for entity in model.get('entities', []):
        if not entity.get('id') or not is_valid_uuid(entity['id']):
            entity['id'] = str(uuid.uuid4())
        # Map both the entity name and normalized name to the entity ID
        entity_name_to_id[entity['name']] = entity['id']
        norm_name = entity['name'].replace(' ', '').replace('_', '').lower()
        entity_name_to_id[norm_name] = entity['id']
        entity.pop('position', None)
        for attr in entity.get('attributes', []):
            if not attr.get('id') or not is_valid_uuid(attr['id']):
                attr['id'] = str(uuid.uuid4())
            attr.pop('classification', None)

    # Fix relationships to use entity IDs instead of names
    for rel in model.get('relationships', []):
        if not rel.get('id') or not is_valid_uuid(rel['id']):
            rel['id'] = str(uuid.uuid4())

        # Map fromEntity to entity ID
        from_entity_name = rel.get('fromEntity', '')
        if from_entity_name in entity_name_to_id:
            rel['fromEntity'] = entity_name_to_id[from_entity_name]
        else:
            # Try normalized name
            norm_from = from_entity_name.replace(' ', '').replace('_', '').lower()
            if norm_from in entity_name_to_id:
                rel['fromEntity'] = entity_name_to_id[norm_from]
            else:
                logger.warning(f"Could not find entity ID for fromEntity: {from_entity_name}")

        # Map toEntity to entity ID
        to_entity_name = rel.get('toEntity', '')
        if to_entity_name in entity_name_to_id:
            rel['toEntity'] = entity_name_to_id[to_entity_name]
        else:
            # Try normalized name
            norm_to = to_entity_name.replace(' ', '').replace('_', '').lower()
            if norm_to in entity_name_to_id:
                rel['toEntity'] = entity_name_to_id[norm_to]
            else:
                logger.warning(f"Could not find entity ID for toEntity: {to_entity_name}")

    return model

async def add_message_to_history(session_id: str, message: Dict[str, Any]) -> None:
    """
    Add a single message to user's chat history.

    Args:
        session_id: User identifier
        message: Message to add

    Raises:
        StorageException: If adding message fails
    """
    try:
        if not session_id:
            raise StorageException("User ID cannot be empty")

        if not isinstance(message, dict):
            raise StorageException("Message must be a dictionary")

        # Get or create history for user
        history = chat_histories.setdefault(session_id, [])
        history.append(message)
        db_history = await db['conversation'].update_one(
                        {"_id": str(session_id)},
                        {
                            "$push": {"messages": message},
                            "$setOnInsert": {"created_at": get_utc_timestamp()},
                            "$set": {"updated_at": get_utc_timestamp()},
                        },
                        upsert=True
                    )

        db_history = await db['conversation'].find_one({"_id": str(session_id)})
        print(db_history['messages'])

        logger.debug(f"Added message to history for user {session_id}. Total messages: {len(history)}")
        logger.info(f"History: {history}")

    except Exception as e:
        logger.error(f"Error adding message to history for user {session_id}: {e}")
        if isinstance(e, StorageException):
            raise
        raise StorageException(f"Failed to add message to history: {str(e)}") from e

async def clear_chat_history(session_id: uuid.UUID) -> None:
    """
    Clear chat history for a specific user.

    Args:
        session_id: User identifier
    """
    try:
        if not session_id:
            logger.warning("Empty session_id provided to clear_chat_history")
            return

        if session_id in chat_histories:
            # del chat_histories[session_id]
            await db['conversation'].delete_one({"_id": str(session_id)})
            logger.info(f"Cleared chat history for user {session_id}")
            # logger.info(f"All chat histories: {chat_histories}")
        else:
            logger.debug(f"No chat history found for user {session_id}")

    except Exception as e:
        logger.error(f"Error clearing chat history for user {session_id}: {e}")

def get_total_users() -> int:
    """
    Get total number of users with chat history.

    Returns:
        Number of users
    """
    try:
        count = len(chat_histories)
        logger.debug(f"Total users with chat history: {count}")
        return count
    except Exception as e:
        logger.error(f"Error getting total users: {e}")
        return 0

def get_total_messages() -> int:
    """
    Get total number of messages across all users.

    Returns:
        Total number of messages
    """
    try:
        total = sum(len(history) for history in chat_histories.values())
        logger.debug(f"Total messages across all users: {total}")
        return total
    except Exception as e:
        logger.error(f"Error getting total messages: {e}")
        return 0

def filter_by_score(data, threshold):
    """
    Given data in the form of:
      [
        {
          'TableName': { metric: score, … },
          'best_match': { metric: [col1, col2, …], … }
        },
        …
      ]
    return a dict mapping each TableName to a dict of
      metric: [score, best_match_columns]
    for all metrics with score > threshold.
    """
    result = {}
    for block in data:
        # each block has exactly one table-key (e.g. 'Customer') plus 'best_match'
        # so extract the table name:
        table = next(k for k in block if k != 'best_match')
        scores = block[table]
        matches = block['best_match']

        # build filtered dict for this table
        filtered = {}
        for metric, score in scores.items():
            if score > threshold:
                # pair score with its best_match column list
                filtered[metric] = [score, matches.get(metric, [])]

        if filtered:
            result[table] = filtered
    return result
