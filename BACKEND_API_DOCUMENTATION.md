# Logical Data Modeling Assistant – Backend API Documentation

## Overview
This backend provides a RESTful API for a Conversational Data Modeling Assistant. Users can interactively design, refine, and query logical data models. The system uses an LLM (Large Language Model) for intent classification and model generation, and supports tool calling for extensibility.

---

## Architecture
- **Framework:** FastAPI (Python)
- **LLM Provider:** OpenAI-compatible API (configurable)
- **Schema Validation:** Pydantic
- **Tool Calling:** Extensible registry for backend tools
- **Logging:** Rotating file and console logging

### System Architecture Diagram

The following diagram illustrates the high-level architecture and data flow of the Logical Data Modeling Assistant backend:

[![temp-Imagedb0-XOe.avif](https://i.postimg.cc/BvJkJXsc/temp-Imagedb0-XOe.avif)](https://postimg.cc/MnrdDZnv)

**Description:**
- The user sends requests to the FastAPI backend.
- The backend classifies intent and, if required, generates a logical data model using an LLM.
- Tool calls, chat history, and logging are managed internally.
- All LLM operations are routed to an OpenAI-compatible API.
- The backend returns structured responses to the user.

---

## Docker Image Size
- The Docker image size is approximately **530MB** (as of the latest build).
- Please consider this image size for deployment planning and resource allocation.

---

## API Endpoints

### 1. `POST /api/v1/model-chat`
**Description:** Main chat endpoint for interacting with the assistant.

**Request Body:**
```json
{
  "query": "string"
}
```

**Response:**
```json
{
  "messages": [
    {
      "role": "user" | "assistant",
      "content": "string | object",
      "timestamp": "ISO 8601 string"
    }
  ]
}
```
- If the user intent is `MODEL`, the assistant returns a logical data model with a `message` field explaining the model or changes.
- If the intent is `CONVO`, the assistant returns a conversational response.

---

### 2. `POST /api/v1/model-chat/reset`
**Description:** Resets the chat history for the user.

**Response:**
```json
{
  "message": "Chat history has been reset."
}
```

---

### 3. `GET /api/v1/model-chat/history`
**Description:** Retrieves the current chat history for the user.

**Response:** Same as `/model-chat` response.

---

## MCP Tool Endpoints

### 4. `POST /api/v1/reference-fibo`
**Description:** MCP Tool endpoint for querying the FIBO (Financial Industry Business Ontology) reference database for relevant information.

**Request Body:**
```json
{
  "query": "string",
  "k": 4,
  "pre_filter": {}
}
```

**Response:**
```json
{
  "query": "string",
  "results": [
    {
      "content": "string",
      "metadata": "object",
      "score": "float"
    }
  ],
  "total_results": "integer"
}
```

**Usage:** This endpoint allows the MCP server or LLM tool orchestration to search the FIBO reference database (FIBORAG vector store) for semantically similar documents and information based on a query string.

**Example:**
```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/reference-fibo' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "BankForInternational Settlements (bank for international settlements)",
    "k": 4,
    "pre_filter": {}
  }'
```

---

## Data Model Schema

**LogicalDataModel:**
```python
class LogicalDataModel(BaseModel):
    id: str
    name: str
    message: str  # Conversational explanation or summary
    entities: List[Entity]
    relationships: List[Relationship]
```
- The `message` field is always required and must summarize the model or changes for the user.

**LogicalPhysicalModel:**
```python
class LogicalPhysicalModel(BaseModel):
    id: str
    name: str
    message: str
    entities: List[LogPhysEntity]
    relationships: List[Relationship]
    useCase: UseCases

class LogPhysEntity(BaseModel):
    id: str
    name: str
    type: str  # "LOGICAL" or "PHYSICAL"
    attributes: List[Attribute]
    tableName: Optional[str] = None  # For PHYSICAL entities
    systemName: Optional[str] = None  # For PHYSICAL entities
    environmentName: Optional[str] = None  # For PHYSICAL entities

class UseCase(BaseModel):
    name: str
    definition: str
    description: str
```
- For PHYSICAL entities, `tableName`, `systemName`, and `environmentName` are populated
- For LOGICAL entities, these fields are set to `null`
- The `useCase` object provides business context for the data model

---

## Tool Calling
- Tools are registered in `core/tools.py`.
- Tool calls are only executed if the LLM returns a response with a `"tool_call"` key.
- All tool executions are logged.

---

## Prompt Engineering
- **SYSTEM_PROMPT:** Instructs the LLM to generate models only when requirements are clear, and always provide a user-friendly summary in the `message` field.
- **INTENT_PROMPT:** Ensures strict classification of user queries as either `MODEL` or `CONVO`.

---

## Environment & Configuration
- **Environment Variables:** (see `.env`)
  - `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_MAX_TOKENS`, etc.
- **Logging:** Logs are written to `logs/app.log` and `logs/error.log` with rotation.

---

## Security & Production Readiness
- No print statements or schema logging in production.
- All sensitive configuration is environment-driven.
- CORS is enabled for all origins by default (can be restricted in production).
- All endpoints are robustly validated and error-handled.

---

## Repository
- [GitHub: 4th-IR/Autonomous-Data-Product-Creation](https://github.com/4th-IR/Autonomous-Data-Product-Creation.git)

---
# Data Movement from ERwin Dumps to LMS

## Overview

This section outlines the technical workflow for moving metadata extracted from **ERwin data model dumps** into a **Language Model Service (LMS)** for indexing and semantic search. The process transforms raw metadata into a structured format suitable for vector storage, enabling advanced AI-powered data discovery.

---

## 1. Source: ERwin Data Dumps

The metadata originates from ERwin model exports (typically CSV or relational extracts), which are parsed into a list of dictionaries named `SQl1`. Each record corresponds to a column and includes associated metadata at the table, environment, and system level.

### Example Fields Extracted:

* **System-level**:

  * `SYSTEM_ID`, `SYSTEM_NAME`
* **Environment-level**:

  * `ENVIRONMENT_ID`, `ENVIRONMENT_NAME`
* **Table-level**:

  * `TABLE_ID`, `TABLE_NAME`, `TABLE_COMMENTS`
* **Column-level**:

  * `COLUMN_NAME`, `COLUMN_DATATYPE`

---

## 2. Data Structuring

To prepare for LMS ingestion, the flat ERwin records are transformed into a hierarchical model using Pydantic schemas.

### Hierarchical Structure:

* **Environment**

  * Contains system and environment identifiers
  * Includes a list of **Tables**

    * Each table includes metadata and a list of **Columns**

###

---

## 3. LMS Integration and Data Ingestion

The structured metadata is sent to the **Language Model Service (LMS)**, which exposes a vector store API to ingest and index data assets for semantic search.

### Endpoint Structure:

```
POST {LMS_API_BASE}/api/v1/vector-store/lms_store/index/QuestSoftware2/add
```

* `LMS_API_BASE`: Base URL of the LMS API
* `lms_store`: Name of the vector store
* `QuestSoftware2`: Index name within the store

### Payload Format (per table):

Each table is pushed as a standalone document with:

* `page_content`: The table name (used as the core vector text)
* `metadata`: A nested structure containing system, environment, table, and column-level details

```json
{
  "page_content": "employees",
  "metadata": {
    "index_name": "QuestSoftware2",
    "metadata": {
      "systemId": 101,
      "systemName": "HR_DB",
      "environmentId": "2",
      "environmentName": "Production",
      "tableName": "employees",
      "tableComments": "Employee details",
      "columnCount": 6,
      "columns": [
        {"name": "emp_id", "datatype": "INT"},
        {"name": "first_name", "datatype": "VARCHAR"}
      ]
    }
  }
}
```

---

## 4. Execution and Error Handling

The `update_lms()` function handles the end-to-end execution:

* Iterates through all environments and tables
* Calls `push_table_to_lms()` for each environment
* Sends the metadata asynchronously using `httpx.AsyncClient`
* Logs progress and handles exceptions per environment

```python
```

---

## 5. Constants and Configuration

| Parameter      | Value               | Purpose                       |
| -------------- | ------------------- | ----------------------------- |
| `store_name`   | "lms_store"        | Target LMS vector store       |
| `index_name`   | "QuestSoftware2"    | Logical grouping of documents |
| `LMS_API_BASE` | Azure container URL | LMS service base URL          |

---

## Conclusion

This pipeline enables automated, structured ingestion of ERwin metadata into an LMS-backed vector store, supporting:

* Centralized metadata indexing
* Semantic discovery of tables and columns
* Scalable and asynchronous data handling

The design ensures future extensibility, including support for custom fields, access control metadata, and dynamic index provisioning. 