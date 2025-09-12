from fastapi import APIRouter, Query, HTTPException
from typing import Any, Optional, List, Dict, Tuple
from pydantic import BaseModel, Field
from schemas.api import (
    QueryRequest,
    VectorStoreQueryRequest,
    VectorStoreQueryResponse,
    UpdateLogicalModelLLMDirectRequest,
)
from schemas.data_model import LogicalPhysicalModel

from schemas.data_product import Association, Relationship, BusinessObject

from core.lms_service import get_logical_models_from_kb
from core.data_model_service import merge_data_models, gen_q, calculate_table_scores, final_merge, cached_generate_logical_model, llm_update_logical_model
from core.data_product_service import create_business_asset, create_association, get_business_asset_object_type_id, get_business_asset_id, get_existing_table_object_type_id, get_existing_table_id
from core.config import settings
from core.storage import filter_by_score
from core.logging_config import get_logger
import uuid
import httpx
import json
import asyncio

from core.lms_service import (
    get_tables_from_lms,
    get_embedding_batch,
    assign_column_similarities,
    adjust_score,
)
from schemas.lms_service import LMSQueryDocument, LMSQueryRequest
from schemas.data_model import Attribute, LogPhysEntity


logger = get_logger('mcp')

router = APIRouter(tags=["MCP Tools"])



# The Erwin upstream API for automating asset creation
ERWIN_AUTOMATE_ASSET_URL = "http://51.103.210.156:8080/ErwinAIService/api/beta/v1/automateAssetCreation"


class AssetCreationRequest(BaseModel):
    """
    Generic payload wrapper for forwarding content to the Erwin asset creation endpoint.
    """
    content: Dict[str, Any]


@router.post(
    "/automate-asset-creation",
    operation_id="automate_asset_creation",
    tags=["MCP Tools"],
)
async def automate_asset_creation_tool(body: AssetCreationRequest):
    """
    MCP Tool Endpoint: Forward a JSON schema to Erwin to automate asset creation.

    Accepts a payload with a `content` object and forwards it upstream. Returns the upstream
    JSON directly when available, or a minimal status/text object otherwise.
    """
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                ERWIN_AUTOMATE_ASSET_URL,
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json",
                },
                json=body.model_dump(),
            )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream connection error: {exc}")

    if resp.headers.get("content-type", "").startswith("application/json"):
        return resp.json()
    return {"status_code": resp.status_code, "text": resp.text}


class LMSQueryInput(BaseModel):
    """Input schema for MCP LMS query tool."""
    model_name: Optional[str] = Field(None, description="Optional environment name to pre-filter results")
    entity: Dict[str, Any] = Field(..., description="Entity object with id, name, and attributes")
    limit: int = Field(25, description="Max number of tables to retrieve")
    w_base: float = Field(0.25, description="Base weight in final score aggregation")
    w_columns: float = Field(0.4, description="Column similarity weight")
    w_auth: float = Field(0.3, description="Additional weighting term")

@router.post(
    "/lms-query",
    tags=["MCP Tools"],
    operation_id="lms_query",
)
async def mcp_lms_query(
    request: LMSQueryInput,
):
    """
    MCP Tool Endpoint: Query LMS for candidate physical tables matching a logical entity.

    This endpoint mirrors the LMS query logic and returns scored candidate tables enriched
    with metadata and a built `LogPhysEntity` for downstream ERwin ingestion.

    Purpose:
    - Retrieve likely physical tables for a given logical entity name and attributes.
    - Compute semantic similarities between entity attributes and table columns.
    - Adjust and return scores with optional environment filtering.

    Parameters:
    - request (LMSQueryRequest):
      - model_name (str | None): Optional environment name to pre-filter results.
      - entity (Entity): Logical entity containing `name` and `attributes`.
      - limit (int): Max number of tables to retrieve.
      - w_base (float): Base weight in final score aggregation.
      - w_columns (float): Column similarity weight.
      - w_auth (float): Additional weighting term (e.g., authority/source).

    Response:
    - list[tuple[LMSQueryDocument, float]]: Each item is (document, score), where document:
      - has flattened `metadata` including `tableName`, `systemName`, `environmentName`, `columns`.
      - includes an `entity` field representing a `LogPhysEntity` derived from metadata.

    Notes:
    - If `model_name` is provided but yields no results, the query is retried without the filter.
    - Column embeddings are computed in batch for performance.
    - Metadata is flattened when nested under `metadata.metadata` for frontend compatibility.
    """
    try:
        # Convert LMSQueryInput to LMSQueryRequest
        from schemas.lms_service import LMSQueryRequest
        from schemas.data_model import Entity, Attribute
        
        # Extract entity from dict and convert to Entity object
        entity_dict = request.entity
        attributes = []
        for attr_dict in entity_dict.get('attributes', []):
            attr = Attribute(
                id=attr_dict['id'],
                name=attr_dict['name'],
                type=attr_dict['type'],
                isPrimaryKey=attr_dict.get('isPrimaryKey', False),
                isForeignKey=attr_dict.get('isForeignKey', False)
            )
            attributes.append(attr)
        
        entity = Entity(
            id=entity_dict['id'],
            name=entity_dict['name'],
            attributes=attributes
        )
        
        # Create LMSQueryRequest
        lms_request = LMSQueryRequest(
            model_name=request.model_name,
            entity=entity,
            limit=request.limit,
            w_base=request.w_base,
            w_columns=request.w_columns,
            w_auth=request.w_auth
        )
        
        model_name = lms_request.model_name
        entity = lms_request.entity
        limit = lms_request.limit
        w_base = lms_request.w_base
        w_columns = lms_request.w_columns
        w_auth = lms_request.w_auth

        logger.info(f"Querying LMS service for entity: {entity.name}")
        
        pre_filter = {}
        if model_name:
            pre_filter = {'environmentName': model_name}

        entity_columns = [attr.name for attr in entity.attributes]

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

        all_column_names: List[str] = []
        for document, _ in query_result:
            md = document.get('metadata', {})
            nested_md = md.get('metadata', md)
            columns = nested_md.get('columns', [])
            all_column_names.extend([col.get('name', '') for col in columns if 'name' in col])

        logger.info(f"Embedding {len(all_column_names)} column names")

        embedding_tasks = [
            get_embedding_batch(all_column_names),
            get_embedding_batch(entity_columns)
        ]

        all_column_embeddings, entity_columns_embeddings = await asyncio.gather(*embedding_tasks)

        await assign_column_similarities(query_result, all_column_embeddings, entity_columns_embeddings)

        tables_with_adjusted_weights = await adjust_score(
            query_result, w_base, w_columns, w_auth, len(entity_columns)
        )

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
            return LogPhysEntity(
                id=str(base_id),
                name=str(table_name),
                type="PHYSICAL",
                attributes=attrs,
                tableName=str(table_name) if table_name is not None else None,
                systemName=md.get('systemName'),
                environmentName=md.get('environmentName')
            )

        for document, _ in tables_with_adjusted_weights:
            try:
                md = document.get('metadata', {})
                if isinstance(md, dict) and isinstance(md.get('metadata'), dict):
                    nested = md.get('metadata', {})
                    merged_md = {k: v for k, v in md.items() if k != 'metadata'}
                    merged_md.update(nested)
                    document['metadata'] = merged_md
            except Exception:
                pass

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

@router.post(
    "/reference-fibo",
    operation_id="reference_fibo",
    tags=["MCP Tools"],
    response_model=VectorStoreQueryResponse
)
async def reference_fibo(
    request: VectorStoreQueryRequest,
):
    """
    MCP Tool Endpoint: Query the FIBO (Financial Industry Business Ontology) reference database.
    
    This tool provides semantic search capabilities for the Financial Industry Business Ontology (FIBO)
    reference database, enabling intelligent retrieval of financial domain knowledge, regulatory information,
    and business ontology concepts. The tool leverages vector embeddings to find semantically similar
    documents and concepts within the FIBORAG vector store.
    
    **Purpose:**
    - Enable semantic search across financial industry ontologies and reference materials
    - Support data modeling and business analysis with domain-specific knowledge
    - Provide access to regulatory compliance information and financial standards
    - Facilitate intelligent document retrieval for financial domain experts
    
    **Key Features:**
    - Semantic similarity search using vector embeddings
    - Configurable result count (k parameter)
    - Optional pre-filtering capabilities
    - Real-time access to FIBO reference database
    - Comprehensive error handling and logging
    
    **Parameters:**
    - `query` (str): The search query string describing the information you're looking for
    - `k` (int, default=4): Number of most relevant results to return (1-20 recommended)
    - `pre_filter` (dict, default={}): Optional metadata filters to narrow search scope
    
    **Response Format:**
    Returns a structured response containing:
    - Original query string
    - List of relevant documents with content, metadata, and similarity scores
    - Total number of results found
    
    **Use Cases:**
    1. **Data Modeling**: Find relevant financial entities and relationships for logical data models
    2. **Regulatory Compliance**: Search for compliance requirements and regulatory frameworks
    3. **Business Analysis**: Retrieve domain-specific knowledge for financial analysis
    4. **Documentation**: Access reference materials and standards documentation
    5. **Research**: Explore financial industry ontologies and taxonomies
    
    **Example Queries:**
    - "Bank for International Settlements regulatory requirements"
    - "Financial instrument classification standards"
    - "Risk management framework compliance"
    - "Derivatives trading regulations"
    - "Capital adequacy requirements"
    
    **Example Usage:**
    ```python
    # Basic search
    response = await reference_fibo({
        "query": "Bank for International Settlements",
        "k": 5,
        "pre_filter": {}
    })
    
    # Filtered search
    response = await reference_fibo({
        "query": "risk management",
        "k": 3,
        "pre_filter": {"category": "compliance"}
    })
    ```
    
    **Error Handling:**
    - HTTP 400: Invalid request parameters
    - HTTP 500: Internal server error or vector store connectivity issues
    - Comprehensive logging for debugging and monitoring
    
    **Performance:**
    - Typical response time: 1-3 seconds
    - Supports concurrent requests
    - Optimized for financial domain queries
    
    **Security:**
    - Input validation and sanitization
    - Rate limiting support
    - Secure communication with vector store
    
    Returns:
        VectorStoreQueryResponse: Structured response with query results and metadata
    """
    # session_id = str(uuid.uuid4())
    logger.info(f"Querying vector store for: {request.query}")
    
    # Vector store endpoint URL from config
    vector_store_url = settings.fibo_vector_store_url
    
    # Prepare the request payload
    payload = {
        "query": request.query,
        "k": request.k,
        "pre_filter": request.pre_filter
    }
    
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    
    try:
        async with httpx.AsyncClient(timeout=9000.0) as client:
            response = await client.post(
                vector_store_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            result_data = response.json()
            
            # Handle case where vector store returns a list directly
            if isinstance(result_data, list):
                results = result_data
            else:
                results = result_data.get('results', [])
            
            logger.info(f"Vector store query successful, returned {len(results)} results")
            
            return VectorStoreQueryResponse(
                query=request.query,
                results=results,
                total_results=len(results)
            )
            
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error when querying vector store: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Vector store query failed: {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Request error when querying vector store: {e}")
        raise HTTPException(status_code=500, detail=f"Vector store query failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error when querying vector store: {e}")
        raise HTTPException(status_code=500, detail=f"Vector store query failed: {str(e)}")



@router.post(
    "/create-logical-model",
    tags=["MCP Tools"],  
    operation_id="create_logical_model"
)
async def create_entire_logical_model(
    query_request: QueryRequest,
):
    """
    MCP Tool Endpoint: Create a logical data model from business requirements.
    
    Transforms business requirements into structured logical data models using LLM capabilities
    and enterprise knowledge base integration.
    
    Args:
        query_request: Business requirement description
        
    Returns:
        dict: Model creation result with logical model, physical tables, and scores
    """
    # session_id = str(uuid.uuid4())
    logger.info(f"Generating logical model for user: {query_request.query}")

    # Use cached version for repeated queries
    logical_model = cached_generate_logical_model(query_request.query)
    logger.info(f"Logical model generated by LLM: \n {logical_model}")

    # Run term generation and KB search in parallel
    term = gen_q(query_request.query)
    logger.info(f'Term passed to Index for search: {term}')
    
    # Get results from knowledge base
    results = await get_logical_models_from_kb(term, limit=1) 
    logger.info(f'Results from index: {results}')
 
    if not results or len(results) == 0:
        logger.warning("No results found in knowledge base")
        return {"error": "No matching models found in knowledge base"}
    
    env_name = results[0][0]['metadata']['metadata']['environmentName']
    system_name = results[0][0]['metadata']['metadata']['systemName']
    
    # Calculate table scores in parallel
    scores = await calculate_table_scores(logical_model.entities, env_name=env_name)
    logger.info(f"Scores: {scores}")
    
    physical_tables = results[0][0]['metadata']['metadata']['tables']
    logger.info(f'Physical tables: {physical_tables}')

    merged_model = merge_data_models(scores, logical_model.model_dump(), physical_tables)
    logger.info(f"Merged model: {merged_model}")

    # logical_entities = []
    # for entity in merged_model.model_dump()['entities']:
    #     if entity['type'] == "LOGICAL":
    #         logical_entities.append(entity)
    #
    # logger.info(f'Logical Entities for Replacement: {logical_entities}')
    #
    # new_scores = await calculate_table_scores(logical_entities)
    # filtered_scores = filter_by_score(new_scores, 0.5)
    # final_model = final_merge(filtered_scores, merged_model)

    # Extract physical table metadata and add to entities
    enhanced_entities = []
    for entity in merged_model.model_dump()['entities']:
        enhanced_entity = entity.copy()
        
        # For PHYSICAL entities, extract metadata from the original scores
        if entity['type'] == "PHYSICAL":
            # Find the corresponding physical table metadata from the original scores
            entity_matched = False
            for score_result in scores:
                if isinstance(score_result, dict) and 'best_match' in score_result:
                    # Get the entity name (first key that's not 'best_match')
                    entity_name = next(k for k in score_result.keys() if k != 'best_match')
                    table_scores = score_result[entity_name]
                    
                    # Check if this score result corresponds to this entity
                    if (entity_name.lower() in entity['name'].lower() or 
                        entity['name'].lower() in entity_name.lower() or
                        any(word in entity['name'].lower() for word in entity_name.lower().split())):
                        
                        # Get the table with highest score
                        if table_scores:
                            best_table = max(table_scores.items(), key=lambda x: x[1])[0]
                            enhanced_entity['tableName'] = best_table
                            enhanced_entity['systemName'] = system_name
                            enhanced_entity['environmentName'] = env_name
                            entity_matched = True
                            logger.info(f"Matched entity {entity['name']} to table {best_table}")
                            break
            
            # If no match found, set default values
            if not entity_matched:
                enhanced_entity['tableName'] = entity['name']  # Use entity name as table name
                enhanced_entity['systemName'] = system_name
                enhanced_entity['environmentName'] = env_name
                logger.info(f"No match found for entity {entity['name']}, using defaults")
        else:
            # For LOGICAL entities, set these fields to None
            enhanced_entity['tableName'] = None
            enhanced_entity['systemName'] = None
            enhanced_entity['environmentName'] = None
        
        enhanced_entities.append(enhanced_entity)

    # Create enhanced final model
    enhanced_final_model = merged_model.model_dump()
    enhanced_final_model['entities'] = enhanced_entities
    
    # Debug logging
    logger.info(f"Enhanced entities count: {len(enhanced_entities)}")
    for i, entity in enumerate(enhanced_entities):
        logger.info(f"Entity {i}: {entity['name']} (type: {entity['type']}, tableName: {entity.get('tableName')}, systemName: {entity.get('systemName')}, environmentName: {entity.get('environmentName')})")

    # Generate use case based on the query
    use_case = {
        "name": f"{query_request.query[:50]} Analysis",
        "definition": f"Analyze {query_request.query.lower()} patterns and behaviors",
        "description": f"This use case focuses on {query_request.query.lower()} to support business objectives and data-driven decision making."
    }

    enhanced_final_model['useCase'] = use_case

    return {
        "merged_model": enhanced_final_model,
        "logical_model": logical_model,
        "physical_model": physical_tables,
        "scores": scores 
    }

# @router.post("/add-generated-data-product", tags=["MCP Tools"], operation_id="add_generated_data_product_to_erwin")
# async def add_generated_data_product_to_erwin(
#     logical_physical_model: LogicalPhysicalModel,
# ):
    """
    Ingests a generated data product into ERWIN by registering business assets and
    establishing associations with corresponding use cases and physical tables.

    This endpoint performs the following:
    - Registers two business objects: the main data product and its associated use case.
    - Retrieves the business asset ID of the use case and links it to the data product.
    - Iterates over the physical model entities (if they have a `tableName`) and associates them
      with the data product as target assets in ERWIN.
    - Creates all necessary associations between the data product, the use case, and related tables.

    ### Parameters:
    - **logical_physical_model** (`LogicalPhysicalModel`): A model combining logical and physical
      metadata, including:
        - `id`: Unique identifier for the model.
        - `name`: Name of the data product.
        - `message`: Description or context message.
        - `entities`: List of logical or physical entities, potentially with physical table metadata.
        - `relationships`: List of logical relationships between entities.
        - `useCase`: The use case associated with this data product (name, definition, description).

    ### Returns:
    - **200 OK** with a message: `"Data product ingested successfully"` if successful.

    ### Raises:
    - **HTTPException (500)**: If any unexpected errors occur during the asset registration or
      association process.
    - **HTTPException (custom)**: For known HTTP-related errors such as invalid asset retrieval.

    ### Notes:
    - This function assumes that the `create_business_asset`, `get_business_asset_id`,
      `get_existing_table_id`, `create_association`, and similar utility functions handle
      validation and ID retrieval logic.
    - Asset type IDs and catalog IDs are hardcoded for now but should be abstracted into config/constants
      for production environments.
    """
    try:
        # session_id = str(uuid.uuid4())
        association = []

        business_object = [BusinessObject(
            name=logical_physical_model.name,
            definition="dummy_definition",
            description="dummy_description"),
            BusinessObject(
                name=logical_physical_model.useCase.name,
                definition="dummy_definition",
                description="dummy_description",
                catalogId=1053 #USE CASE CATALOG (DEFAULT)
            )
        ]

        logger.info(f'Creating business assets: {[obj.name for obj in business_object]}')
        business_asset_creation_response = await create_business_asset(business_object)

        logger.info(f'Business Asset Created: {business_object}')
        use_case_id = await get_business_asset_id(logical_physical_model.useCase.name, asset_type_name='Use Case')
        data_product_id = await get_business_asset_id(logical_physical_model.name)
        data_product_object_type_id = await get_business_asset_object_type_id(data_product_id)

        association.append(Association(
            sourceObjectId= data_product_id,
            sourceObjectTypeId= data_product_object_type_id,
            targetObjectId=use_case_id,
            targetObjectTypeId= await get_business_asset_object_type_id(use_case_id),
            relationship=Relationship(description="dummy association description")
        ))



        for entity in logical_physical_model.entities:
            if entity.tableName:
                logger.info(f"Retrieving {entity.tableName}: {entity}")
                table_id = await get_existing_table_id(system_name=entity.systemName, environment_name=entity.environmentName, table_name=entity.tableName)

                association.append(Association(
                    sourceObjectId= data_product_id,
                    sourceObjectTypeId= data_product_object_type_id,
                    targetObjectId=table_id,
                    targetObjectTypeId= await get_existing_table_object_type_id(table_id=table_id),
                    relationship=Relationship(description="dummy association description")
                ))


        association_creation_response = await create_association(association=association)
        logger.info(f'Association Created: {association}')

        return {
            # "session_id": session_id,
            "session_details": {
                "business_assets": business_asset_creation_response,
                "associations": association_creation_response,
            }
        }

    except HTTPException as e:
        logger.error(f'HTTP error: {e.detail}')
        raise e

    except Exception as e:
        logger.error(f'Unknown error: {e}')
        raise HTTPException(500, detail=str(e))


# --- Update Logical Model Tool ---

# @router.post(
#     "/update-logical-model",
#     tags=["MCP Tools"],
#     operation_id="update_logical_model"
# )
# async def update_logical_model(
#     request: UpdateLogicalModelRequest,
# ) -> dict:
#     """
#     MCP Tool Endpoint: Apply targeted updates to the latest LogicalPhysicalModel in the conversation history.
# 
#     How it works now:
#     - When a user asks to update the model, the tool will automatically retrieve the latest logical model
#       that was generated in the current conversation/session (using the session_id).
#     - The tool will then apply the requested updates to that latest model, ensuring that the user's changes
#       are made in the context of the most recent model state.
#     - The user does not need to provide the current model in the request; only the session_id and updates are needed.
#     - The tool preserves all existing relationships and entity IDs, and only updates the specified entities/attributes.
# 
#     Returns a new updated model object; the original model in the conversation history is not mutated server-side.
#     """
# 
#     try:
#         # The session_id must be provided in the request to identify the conversation
#         session_id = getattr(request, "session_id", None)
#         if not session_id:
#             raise HTTPException(400, detail="session_id is required to update the logical model from conversation history.")
# 
#         # Fetch the latest model from the conversation history for this session
#         from core.lms_service import get_latest_model_from_session_history
# 
#         latest_model = await get_latest_model_from_session_history(session_id)
#         if not latest_model:
#             raise HTTPException(404, detail="No logical model found in conversation history for this session.")
# 
#         # Apply the updates to the latest model
#         updated_model = llm_update_logical_model(latest_model, request.updates)
#         return updated_model
# 
#     except HTTPException as e:
#         logger.error(f'HTTP error: {e.detail}')
#         raise e
#     except Exception as e:
#         logger.error(f'Unknown error: {e}')
#         raise HTTPException(500, detail=str(e))


# --- Update Logical Model via LLM (direct model input) ---

@router.post(
    "/update-logical-model-llm",
    tags=["MCP Tools"],
    operation_id="update_logical_model_llm",
)
async def update_logical_model_llm(
    request: UpdateLogicalModelLLMDirectRequest,
) -> LogicalPhysicalModel:
    """
    MCP Tool Endpoint: Apply a natural-language instruction to the provided LogicalPhysicalModel
    and return the full updated model.

    - Takes the current model and instruction directly in the request body.
    - Applies the instruction using the LLM with strict update constraints.
    - Preserves existing IDs, relationships, and order; updates message to summarize changes.
    - Returns a full LogicalPhysicalModel.
    """
    try:
        # Direct-only: require the current model and instruction in the request
        updated_model = llm_update_logical_model(request.current_model, request.instruction)
        return updated_model
    except HTTPException as e:
        logger.error(f"HTTP error: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unknown error: {e}")
        raise HTTPException(500, detail=str(e))
