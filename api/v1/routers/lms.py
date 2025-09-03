from core.lms_service import (
    get_tables_from_lms,
    get_embedding,
    adjust_score,
    assign_column_similarities,
    assign_environment_similarities,
)
from core.logging_config import get_logger
from fastapi import APIRouter, HTTPException, status, Depends
from schemas.data_model import Entity, Attribute, LogPhysEntity
from typing import List, Tuple, Dict, Any
from core.lms_service import get_embedding_batch
from schemas.lms_service import LMSQueryDocument, LMSQueryRequest
from core.authentication.auth_middleware import get_current_token
from schemas.token import TokenData
import asyncio

logger = get_logger(__name__)
router = APIRouter(tags=["LMS Service"])


@router.post("/lms-query", response_model=list[tuple[LMSQueryDocument, float]])
async def lms_query(
    request: LMSQueryRequest
    # current_user: TokenData = Depends(get_current_token)
) -> list[tuple[LMSQueryDocument, float]]:
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

        # Step 2: Fetch candidate tables from LMS (with fallback if no results)
        query_result: List[Tuple[LMSQueryDocument, float]] = await get_tables_from_lms(
            entity.name, limit=limit, pre_filter=pre_filter
        )

        if len(query_result) == 0 and pre_filter:
            logger.info(
                f"No tables with pre_filter for entity: {entity.name} and model: {model_name}. Retrying without pre_filter."
            )
            query_result = await get_tables_from_lms(entity.name, limit=limit, pre_filter={})

        logger.info(f"{len(query_result)} tables returned from LMS service")

        if len(query_result) == 0:
            logger.info(
                f"No tables returned from LMS service with name: {entity.name}"
            )
            return []

        # Collect all column names
        all_column_names = []
        for document, _ in query_result:
            # Handle both nested and flattened metadata shapes
            md = document.get('metadata', {})
            nested_md = md.get('metadata', md)
            columns = nested_md.get('columns', [])
            all_column_names.extend([col.get('name', '') for col in columns if 'name' in col])

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

        # Helper to construct an Entity from LMS metadata (physical table -> Entity)
        def build_entity_from_metadata(doc: dict) -> LogPhysEntity:
            md = doc.get('metadata', {})
            table_name = md.get('tableName', 'Unknown')
            cols = md.get('columns', [])
            attrs: list[Attribute] = []
            base_id = doc.get('id', table_name)
            for idx, col in enumerate(cols, start=1):
                attr = Attribute(
                    id=f"{base_id}.{idx}",
                    name=col.get('name', f'col_{idx}'),
                    type=(col.get('datatype') or 'STRING'),
                    isPrimaryKey=False,
                    isForeignKey=False,
                )
                attrs.append(attr)
            # Build LogPhysEntity enriched with system/environment metadata
            return LogPhysEntity(
                id=str(base_id),
                name=str(table_name),
                type="PHYSICAL",
                attributes=attrs,
                tableName=str(table_name) if table_name is not None else None,
                systemName=md.get('systemName'),
                environmentName=md.get('environmentName')
            )

        # Normalize documents for frontend compatibility: flatten nested metadata and attach matched entity
        for document, _ in tables_with_adjusted_weights:
            # Flatten metadata if nested under 'metadata'
            try:
                md = document.get('metadata', {})
                if isinstance(md, dict) and isinstance(md.get('metadata'), dict):
                    nested = md.get('metadata', {})
                    # Merge top-level md (excluding the nested key) with nested
                    merged_md = {k: v for k, v in md.items() if k != 'metadata'}
                    merged_md.update(nested)
                    document['metadata'] = merged_md
            except Exception:
                pass

            # Attach matched physical entity object (for ERwin ingestion)
            try:
                matched_entity = build_entity_from_metadata(document)
                document['entity'] = matched_entity.model_dump()
            except Exception:
                try:
                    document['entity'] = build_entity_from_metadata(document).dict()
                except Exception:
                    document['entity'] = None

        return tables_with_adjusted_weights

    except HTTPException as e:
        logger.error(f"HTTPException during LMS query: {e}")
        raise e

    except Exception as e:
        logger.error(f"Unexpected error during LMS query: {e}")
        raise HTTPException(status_code=500, detail=str(e))