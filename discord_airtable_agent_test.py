import os
import json
import difflib
import requests
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import discord
from discord.ext import commands

# Import LangChain components.
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferWindowMemory  # Using windowed memory to limit context
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.tools import StructuredTool
from langchain_openai import ChatOpenAI

# =============================================================================
# Load Environment Variables
# =============================================================================

load_dotenv()

OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_CLIENT_SECRET = os.getenv("AIRTABLE_CLIENT_SECRET")  # May be None
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

BASE_URL = "https://api.airtable.com/v0"
META_URL = "https://api.airtable.com/v0/meta"
CONTENT_URL = "https://content.airtable.com/v0"

# =============================================================================
# Global Variables for Resolved Base and Table Information
# =============================================================================

RESOLVED_BASE_ID = None  # Once a base is resolved, store its ID here.
RESOLVED_TABLE_NAMES = {}  # Dictionary mapping base_id -> {input_table_name: resolved_table_name}

# =============================================================================
# Helper Functions for Base and Table Resolution
# =============================================================================

def resolve_base_id(base_identifier: str) -> Optional[str]:
    """
    If base_identifier starts with 'app', assume it is already a base ID.
    Otherwise, try to resolve it by matching the provided base name.
    Once resolved, store it in RESOLVED_BASE_ID.
    """
    global RESOLVED_BASE_ID
    if base_identifier.startswith("app"):
        RESOLVED_BASE_ID = base_identifier
        return base_identifier
    # If already resolved, simply return it.
    if RESOLVED_BASE_ID and base_identifier.lower() in RESOLVED_BASE_ID.lower():
        return RESOLVED_BASE_ID
    url = f"{META_URL}/bases"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    if AIRTABLE_CLIENT_SECRET:
        headers["X-Airtable-Client-Secret"] = AIRTABLE_CLIENT_SECRET
    response = requests.get(url, headers=headers)
    if response.ok:
        bases = response.json().get("bases", [])
        candidate_names = [base["name"] for base in bases]
        matches = difflib.get_close_matches(base_identifier, candidate_names, n=1, cutoff=0.6)
        if matches:
            for base in bases:
                if base["name"].lower() == matches[0].lower():
                    RESOLVED_BASE_ID = base["id"]
                    return base["id"]
    return None

def resolve_table_name(base_id: str, table_identifier: str) -> Optional[str]:
    """
    Uses fuzzy matching to resolve a table name and stores the result in RESOLVED_TABLE_NAMES.
    """
    global RESOLVED_TABLE_NAMES
    if base_id not in RESOLVED_TABLE_NAMES:
        RESOLVED_TABLE_NAMES[base_id] = {}
    if table_identifier in RESOLVED_TABLE_NAMES[base_id]:
        return RESOLVED_TABLE_NAMES[base_id][table_identifier]
    url = f"{META_URL}/bases/{base_id}/tables"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    if AIRTABLE_CLIENT_SECRET:
        headers["X-Airtable-Client-Secret"] = AIRTABLE_CLIENT_SECRET
    response = requests.get(url, headers=headers)
    if response.ok:
        tables = response.json().get("tables", [])
        candidate_names = [table.get("name", "") for table in tables]
        matches = difflib.get_close_matches(table_identifier, candidate_names, n=1, cutoff=0.6)
        if matches:
            RESOLVED_TABLE_NAMES[base_id][table_identifier] = matches[0]
            return matches[0]
    return None

def resolve_table_id(base_id: str, table_identifier: str) -> Optional[str]:
    """
    Resolves the table ID for a given base and table name.
    """
    url = f"{META_URL}/bases/{base_id}/tables"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    if AIRTABLE_CLIENT_SECRET:
        headers["X-Airtable-Client-Secret"] = AIRTABLE_CLIENT_SECRET
    response = requests.get(url, headers=headers)
    if response.ok:
        tables = response.json().get("tables", [])
        candidate_names = [table.get("name", "") for table in tables]
        matches = difflib.get_close_matches(table_identifier, candidate_names, n=1, cutoff=0.6)
        if matches:
            for table in tables:
                if table.get("name", "").lower() == matches[0].lower():
                    return table.get("id")
    return None

# =============================================================================
# Define Pydantic Models for Structured Input
# =============================================================================

class EmptyInput(BaseModel):
    pass

class GetBaseSchemaInput(BaseModel):
    base_id: str

class ListRecordsInput(BaseModel):
    base_id: str
    table_name: str
    fields: Optional[List[str]] = None

class CreateRecordInput(BaseModel):
    base_id: str
    table_name: str
    fields: dict

class UpdateRecordInput(BaseModel):
    base_id: str
    table_name: str
    record_id: str
    fields: dict

class DeleteRecordInput(BaseModel):
    base_id: str
    table_name: str
    record_id: str

class GetRecordInput(BaseModel):
    base_id: str
    table_name: str
    record_id: str
    fields: Optional[List[str]] = None

class FilterRecordsInput(BaseModel):
    base_id: str
    table_name: str
    filter_field: str
    filter_value: str
    fields: Optional[List[str]] = None

# New models for bulk operations and other endpoints

class UpdateMultipleRecordsInput(BaseModel):
    base_id: str
    table_name: str
    records: List[dict]
    performUpsert: Optional[dict] = None

class CreateRecordsInput(BaseModel):
    base_id: str
    table_name: str
    records: List[dict]

class DeleteMultipleRecordsInput(BaseModel):
    base_id: str
    table_name: str
    record_ids: List[str]

class SyncCSVInput(BaseModel):
    base_id: str
    table_name: str
    sync_id: str
    csv_data: str

class UploadAttachmentInput(BaseModel):
    base_id: str
    record_id: str
    attachment_field: str
    content_type: str
    file: str
    filename: str

class UpdateFieldInput(BaseModel):
    base_id: str
    table_id: str
    field_id: str
    name: Optional[str] = None
    description: Optional[str] = None

class CreateFieldInput(BaseModel):
    base_id: str
    table_id: str
    name: str
    description: Optional[str] = None
    options: Optional[dict] = None
    type: str

class UpdateTableInput(BaseModel):
    base_id: str
    table_id_or_name: str
    name: Optional[str] = None
    description: Optional[str] = None

class CreateTableInput(BaseModel):
    base_id: str
    name: str
    description: Optional[str] = None
    fields: List[dict]

class CreateBaseInput(BaseModel):
    name: str
    tables: List[dict]
    workspace_id: str

class ListViewsInput(BaseModel):
    base_id: str

class DeleteViewInput(BaseModel):
    base_id: str
    view_id: str

class GetTableSchemaInput(BaseModel):
    base_id: str
    table_name: str

# =============================================================================
# Define Airtable API Tool Functions (Minimal Outputs, Reuse Resolved IDs)
# =============================================================================

def list_bases_tool(**kwargs) -> str:
    global RESOLVED_BASE_ID
    if RESOLVED_BASE_ID:
        return f"Using resolved base: {RESOLVED_BASE_ID}"
    url = f"{META_URL}/bases"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    if AIRTABLE_CLIENT_SECRET:
        headers["X-Airtable-Client-Secret"] = AIRTABLE_CLIENT_SECRET
    response = requests.get(url, headers=headers)
    if response.ok:
        bases = response.json().get("bases", [])
        if not bases:
            return "No bases found."
        minimal_list = "\n".join([f"{base['name']} (ID: {base['id']})" for base in bases])
        return f"Bases:\n{minimal_list}"
    return f"Error listing bases: {response.status_code} {response.text}"

def get_base_schema_tool(*, base_id: str) -> str:
    resolved_base = resolve_base_id(base_id)
    if not resolved_base:
        return f"Could not resolve base '{base_id}'."
    base_id = resolved_base
    url = f"{META_URL}/bases/{base_id}/tables"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    if AIRTABLE_CLIENT_SECRET:
        headers["X-Airtable-Client-Secret"] = AIRTABLE_CLIENT_SECRET
    response = requests.get(url, headers=headers)
    if response.ok:
        tables = response.json().get("tables", [])
        if not tables:
            return f"No tables in base {base_id}."
        minimal = "\n".join([f"{table['name']} (ID: {table['id']})" for table in tables])
        return f"Tables in base {base_id}:\n{minimal}"
    return f"Error retrieving schema: {response.status_code} {response.text}"

def get_table_schema_tool(*, base_id: str, table_name: str) -> str:
    base_id = resolve_base_id(base_id) or base_id
    resolved = resolve_table_name(base_id, table_name)
    if not resolved:
        return f"Could not resolve a table matching '{table_name}'."
    url = f"{META_URL}/bases/{base_id}/tables"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    if AIRTABLE_CLIENT_SECRET:
        headers["X-Airtable-Client-Secret"] = AIRTABLE_CLIENT_SECRET
    response = requests.get(url, headers=headers)
    if response.ok:
        for table in response.json().get("tables", []):
            if table.get("name", "").lower() == resolved.lower():
                fields = table.get("fields", [])
                field_names = [field.get("name", "") for field in fields]
                return ", ".join(field_names)
        return f"Table '{resolved}' not found in base {base_id}."
    return f"Error retrieving table schema: {response.status_code} {response.text}"

def list_records_tool(*, base_id: str, table_name: str, fields: Optional[List[str]] = None) -> str:
    base_id = resolve_base_id(base_id) or base_id
    resolved = resolve_table_name(base_id, table_name)
    if resolved:
        table_name = resolved
    url = f"{BASE_URL}/{base_id}/{table_name}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    params = {}
    if fields:
        params = [('fields[]', f) for f in fields]
    response = requests.get(url, headers=headers, params=params)
    if response.ok:
        records = response.json().get("records", [])
        if not records:
            return f"No records in '{table_name}'."
        result_lines = []
        for record in records:
            rec_id = record.get("id")
            if fields:
                rec_fields = record.get("fields", {})
                filtered = {k: rec_fields.get(k) for k in fields}
                result_lines.append(f"{rec_id}: {filtered}")
            else:
                result_lines.append(f"{rec_id}")
        return "\n".join(result_lines)
    return f"Error listing records: {response.status_code} {response.text}"

def create_record_tool(*, base_id: str, table_name: str, fields: dict) -> str:
    base_id = resolve_base_id(base_id) or base_id
    resolved = resolve_table_name(base_id, table_name)
    if resolved:
        table_name = resolved
    url = f"{BASE_URL}/{base_id}/{table_name}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}", "Content-Type": "application/json"}
    payload = {"fields": fields}
    response = requests.post(url, headers=headers, json=payload)
    if response.ok:
        rec_id = response.json().get("id", "Unknown")
        return f"Created record {rec_id}."
    return f"Error creating record: {response.status_code} {response.text}"

def update_record_tool(*, base_id: str, table_name: str, record_id: str, fields: dict) -> str:
    base_id = resolve_base_id(base_id) or base_id
    resolved = resolve_table_name(base_id, table_name)
    if resolved:
        table_name = resolved
    url = f"{BASE_URL}/{base_id}/{table_name}/{record_id}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}", "Content-Type": "application/json"}
    payload = {"fields": fields}
    response = requests.patch(url, headers=headers, json=payload)
    if response.ok:
        return f"Updated record {record_id}."
    return f"Error updating record: {response.status_code} {response.text}"

def delete_record_tool(*, base_id: str, table_name: str, record_id: str) -> str:
    base_id = resolve_base_id(base_id) or base_id
    resolved = resolve_table_name(base_id, table_name)
    if resolved:
        table_name = resolved
    url = f"{BASE_URL}/{base_id}/{table_name}/{record_id}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    response = requests.delete(url, headers=headers)
    if response.ok:
        return f"Deleted record {record_id}."
    return f"Error deleting record: {response.status_code} {response.text}"

def get_record_tool(*, base_id: str, table_name: str, record_id: str, fields: Optional[List[str]] = None) -> str:
    base_id = resolve_base_id(base_id) or base_id
    resolved = resolve_table_name(base_id, table_name)
    if resolved:
        table_name = resolved
    url = f"{BASE_URL}/{base_id}/{table_name}/{record_id}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    params = {}
    if fields:
        params = [('fields[]', f) for f in fields]
    response = requests.get(url, headers=headers, params=params)
    if response.ok:
        data = response.json()
        rec_id = data.get("id")
        if fields:
            rec_fields = data.get("fields", {})
            filtered = {k: rec_fields.get(k) for k in fields}
            return f"{rec_id}: {filtered}"
        else:
            return f"{rec_id}"
    return f"Error getting record: {response.status_code} {response.text}"

def filter_records_tool(*, base_id: str, table_name: str, filter_field: str, filter_value: str, fields: Optional[List[str]] = None) -> str:
    base_id = resolve_base_id(base_id) or base_id
    resolved = resolve_table_name(base_id, table_name)
    if resolved:
        table_name = resolved
    url = f"{BASE_URL}/{base_id}/{table_name}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    safe_value = filter_value.replace("'", "\\'")
    filter_formula = f"{{{filter_field}}}='{safe_value}'"
    params = {"filterByFormula": filter_formula}
    if fields:
        params_list = list(params.items()) + [('fields[]', f) for f in fields]
    else:
        params_list = list(params.items())
    response = requests.get(url, headers=headers, params=params_list)
    if response.status_code == 422:
        error_msg = response.json().get("error", {}).get("message", "")
        if "Unknown field names" in error_msg:
            available_fields = get_table_schema_tool(base_id=base_id, table_name=table_name)
            return (f"Error: The field '{filter_field}' is not recognized. "
                    f"Available fields are: {available_fields}. "
                    "Please specify which field to use for filtering.")
    if response.ok:
        records = response.json().get("records", [])
        if not records:
            return f"No records found using filter {filter_formula}."
        if len(records) > 1:
            table_id = resolve_table_id(base_id, table_name) or "UNKNOWN_TABLE_ID"
            links = []
            for record in records:
                rec_id = record.get("id")
                link = f"https://airtable.com/{base_id}/{table_id}/{rec_id}"
                links.append(link)
            return ("Multiple records found. Please review the links below and specify which record to use:\n" +
                    "\n".join(links))
        record = records[0]
        rec_id = record.get("id")
        if fields:
            rec_fields = record.get("fields", {})
            filtered = {k: rec_fields.get(k) for k in fields}
            return f"{rec_id}: {filtered}"
        else:
            return f"{rec_id}"
    return f"Error filtering records: {response.status_code} {response.text}"

def update_multiple_records_tool(*, base_id: str, table_name: str, records: List[dict], performUpsert: Optional[dict] = None) -> str:
    base_id = resolve_base_id(base_id) or base_id
    resolved = resolve_table_name(base_id, table_name)
    if resolved:
        table_name = resolved
    url = f"{BASE_URL}/{base_id}/{table_name}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}", "Content-Type": "application/json"}
    payload = {"records": records}
    if performUpsert:
        payload["performUpsert"] = performUpsert
    response = requests.patch(url, headers=headers, json=payload)
    if response.ok:
        return "Bulk update successful."
    return f"Error in bulk update: {response.status_code} {response.text}"

def create_records_tool(*, base_id: str, table_name: str, records: List[dict]) -> str:
    base_id = resolve_base_id(base_id) or base_id
    resolved = resolve_table_name(base_id, table_name)
    if resolved:
        table_name = resolved
    url = f"{BASE_URL}/{base_id}/{table_name}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}", "Content-Type": "application/json"}
    payload = {"records": records}
    response = requests.post(url, headers=headers, json=payload)
    if response.ok:
        return "Bulk creation successful."
    return f"Error in bulk creation: {response.status_code} {response.text}"

def delete_multiple_records_tool(*, base_id: str, table_name: str, record_ids: List[str]) -> str:
    base_id = resolve_base_id(base_id) or base_id
    resolved = resolve_table_name(base_id, table_name)
    if resolved:
        table_name = resolved
    url = f"{BASE_URL}/{base_id}/{table_name}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    params = [('records[]', rid) for rid in record_ids]
    response = requests.delete(url, headers=headers, params=params)
    if response.ok:
        return "Bulk deletion successful."
    return f"Error in bulk deletion: {response.status_code} {response.text}"

def sync_csv_tool(*, base_id: str, table_name: str, sync_id: str, csv_data: str) -> str:
    base_id = resolve_base_id(base_id) or base_id
    resolved = resolve_table_name(base_id, table_name)
    if resolved:
        table_name = resolved
    url = f"{BASE_URL}/{base_id}/{table_name}/sync/{sync_id}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}", "Content-Type": "text/csv"}
    response = requests.post(url, headers=headers, data=csv_data)
    if response.ok:
        return "CSV sync successful."
    return f"Error syncing CSV: {response.status_code} {response.text}"

def upload_attachment_tool(*, base_id: str, record_id: str, attachment_field: str, content_type: str, file: str, filename: str) -> str:
    base_id = resolve_base_id(base_id) or base_id
    url = f"{CONTENT_URL}/{base_id}/{record_id}/{attachment_field}/uploadAttachment"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}", "Content-Type": "application/json"}
    payload = {"contentType": content_type, "file": file, "filename": filename}
    response = requests.post(url, headers=headers, json=payload)
    if response.ok:
        return "Attachment uploaded successfully."
    return f"Error uploading attachment: {response.status_code} {response.text}"

def update_field_tool(*, base_id: str, table_id: str, field_id: str, name: Optional[str] = None, description: Optional[str] = None) -> str:
    base_id = resolve_base_id(base_id) or base_id
    url = f"{META_URL}/bases/{base_id}/tables/{table_id}/fields/{field_id}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}", "Content-Type": "application/json"}
    payload = {}
    if name:
        payload["name"] = name
    if description:
        payload["description"] = description
    response = requests.patch(url, headers=headers, json=payload)
    if response.ok:
        return "Field updated successfully."
    return f"Error updating field: {response.status_code} {response.text}"

def create_field_tool(*, base_id: str, table_id: str, name: str, type: str, description: Optional[str] = None, options: Optional[dict] = None) -> str:
    base_id = resolve_base_id(base_id) or base_id
    url = f"{META_URL}/bases/{base_id}/tables/{table_id}/fields"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}", "Content-Type": "application/json"}
    payload = {"name": name, "type": type}
    if description:
        payload["description"] = description
    if options:
        payload["options"] = options
    response = requests.post(url, headers=headers, json=payload)
    if response.ok:
        return "Field created successfully."
    return f"Error creating field: {response.status_code} {response.text}"

def update_table_tool(*, base_id: str, table_id_or_name: str, name: Optional[str] = None, description: Optional[str] = None) -> str:
    base_id = resolve_base_id(base_id) or base_id
    url = f"{META_URL}/bases/{base_id}/tables/{table_id_or_name}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}", "Content-Type": "application/json"}
    payload = {}
    if name:
        payload["name"] = name
    if description:
        payload["description"] = description
    response = requests.patch(url, headers=headers, json=payload)
    if response.ok:
        return "Table updated successfully."
    return f"Error updating table: {response.status_code} {response.text}"

def create_table_tool(*, base_id: str, name: str, fields: List[dict], description: Optional[str] = None) -> str:
    base_id = resolve_base_id(base_id) or base_id
    url = f"{META_URL}/bases/{base_id}/tables"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}", "Content-Type": "application/json"}
    payload = {"name": name, "fields": fields}
    if description:
        payload["description"] = description
    response = requests.post(url, headers=headers, json=payload)
    if response.ok:
        return "Table created successfully."
    return f"Error creating table: {response.status_code} {response.text}"

def create_base_tool(*, name: str, tables: List[dict], workspace_id: str) -> str:
    url = f"{META_URL}/bases"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}", "Content-Type": "application/json"}
    payload = {"name": name, "tables": tables, "workspaceId": workspace_id}
    response = requests.post(url, headers=headers, json=payload)
    if response.ok:
        return "Base created successfully."
    return f"Error creating base: {response.status_code} {response.text}"

def list_views_tool(*, base_id: str) -> str:
    base_id = resolve_base_id(base_id) or base_id
    url = f"{META_URL}/bases/{base_id}/views"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    response = requests.get(url, headers=headers)
    if response.ok:
        views = response.json()
        return f"Views: {views}"
    return f"Error listing views: {response.status_code} {response.text}"

def delete_view_tool(*, base_id: str, view_id: str) -> str:
    base_id = resolve_base_id(base_id) or base_id
    url = f"{META_URL}/bases/{base_id}/views/{view_id}"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    response = requests.delete(url, headers=headers)
    if response.ok:
        return "View deleted successfully."
    return f"Error deleting view: {response.status_code} {response.text}"

# =============================================================================
# Configure Agent Tools (Using StructuredTool)
# =============================================================================

tools = [
    StructuredTool(
        name="List Bases",
        func=list_bases_tool,
        args_schema=EmptyInput,
        description="Lists all Airtable bases (ID and name only) using minimal output."
    ),
    StructuredTool(
        name="Get Base Schema",
        func=get_base_schema_tool,
        args_schema=GetBaseSchemaInput,
        description="Lists the tables (ID and name) in a base."
    ),
    StructuredTool(
        name="Get Table Schema",
        func=get_table_schema_tool,
        args_schema=GetTableSchemaInput,
        description="Returns a comma-separated list of field names in a table."
    ),
    StructuredTool(
        name="List Records",
        func=list_records_tool,
        args_schema=ListRecordsInput,
        description="Lists records in a table. Returns only record IDs (or specified fields)."
    ),
    StructuredTool(
        name="Create Record",
        func=create_record_tool,
        args_schema=CreateRecordInput,
        description="Creates a record with the specified fields."
    ),
    StructuredTool(
        name="Update Record",
        func=update_record_tool,
        args_schema=UpdateRecordInput,
        description="Updates a record's fields."
    ),
    StructuredTool(
        name="Delete Record",
        func=delete_record_tool,
        args_schema=DeleteRecordInput,
        description="Deletes a record by ID."
    ),
    StructuredTool(
        name="Get Record",
        func=get_record_tool,
        args_schema=GetRecordInput,
        description="Retrieves a record. Returns only requested fields (or just the ID if none specified)."
    ),
    StructuredTool(
        name="Filter Records",
        func=filter_records_tool,
        args_schema=FilterRecordsInput,
        description=(
            "Filters records by a given field condition. If the filter field is unknown, returns available fields "
            "and asks for clarification. If multiple records match, returns clickable links for each record."
        )
    ),
    StructuredTool(
        name="Update Multiple Records",
        func=update_multiple_records_tool,
        args_schema=UpdateMultipleRecordsInput,
        description="Performs bulk update of records."
    ),
    StructuredTool(
        name="Create Records",
        func=create_records_tool,
        args_schema=CreateRecordsInput,
        description="Creates multiple records at once."
    ),
    StructuredTool(
        name="Delete Multiple Records",
        func=delete_multiple_records_tool,
        args_schema=DeleteMultipleRecordsInput,
        description="Deletes multiple records specified by their IDs."
    ),
    StructuredTool(
        name="Sync CSV Data",
        func=sync_csv_tool,
        args_schema=SyncCSVInput,
        description="Syncs CSV data to a table; only the CSV string is sent."
    ),
    StructuredTool(
        name="Upload Attachment",
        func=upload_attachment_tool,
        args_schema=UploadAttachmentInput,
        description="Uploads a base64-encoded file as an attachment to a record."
    ),
    StructuredTool(
        name="Update Field",
        func=update_field_tool,
        args_schema=UpdateFieldInput,
        description="Updates a field's configuration."
    ),
    StructuredTool(
        name="Create Field",
        func=create_field_tool,
        args_schema=CreateFieldInput,
        description="Creates a new field in a table."
    ),
    StructuredTool(
        name="Update Table",
        func=update_table_tool,
        args_schema=UpdateTableInput,
        description="Updates a table's name or description."
    ),
    StructuredTool(
        name="Create Table",
        func=create_table_tool,
        args_schema=CreateTableInput,
        description="Creates a new table with specified fields."
    ),
    StructuredTool(
        name="Create Base",
        func=create_base_tool,
        args_schema=CreateBaseInput,
        description="Creates a new base with one or more tables."
    ),
    StructuredTool(
        name="List Views",
        func=list_views_tool,
        args_schema=ListViewsInput,
        description="Lists views in a base."
    ),
    StructuredTool(
        name="Delete View",
        func=delete_view_tool,
        args_schema=DeleteViewInput,
        description="Deletes a view by its ID."
    ),
]

# =============================================================================
# Set Up the Agent with Limited Memory to Prevent Context Overload
# =============================================================================

prompt = hub.pull("hwchase17/structured-chat-agent")

llm = ChatOpenAI(
    openai_api_key=OPEN_AI_API_KEY,
    model="gpt-4"
)

# Use a windowed memory (only last 3 messages) to keep context minimal.
memory = ConversationBufferWindowMemory(memory_key="chat_history", k=3, return_messages=True)

initial_message = (
    "You are an AI assistant for Airtable. Your job is to help users interact with Airtable's API. "
    "If the user does not provide an exact base ID or table name, always ask for clarification. "
    "When a base or table is provided, use fuzzy matching to resolve and cache the corresponding IDs/names, "
    "and then use them in all subsequent calls without repeating full listings. "
    "Return only minimal responses (e.g., 'Using resolved base: appMxsw3zihH6FLoi') once the information is resolved. "
    "When filtering records, return only the record IDs or only the specific fields requested. "
    "If a filter error occurs due to an unknown field, use 'Get Table Schema' to show available fields and ask for clarification. "
    "If multiple records match, return clickable links in the format:\n"
    "https://airtable.com/{base_id}/{table_id}/{record_id}\n"
    "and ask the user to specify which record to use."
)
memory.chat_memory.add_message(SystemMessage(content=initial_message))

agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
)

# =============================================================================
# Discord Bot Integration
# =============================================================================

bot = commands.Bot(command_prefix="!")

@bot.event
async def on_ready():
    print(f"{bot.user} is now online!")

@bot.command(name="airtable")
async def airtable(ctx, *, query: str):
    """
    This command passes the user's query to the Airtable agent and sends the response.
    Usage: !airtable <your query here>
    """
    loop = asyncio.get_running_loop()
    # Run the agent's invoke method in an executor to avoid blocking the Discord event loop.
    response = await loop.run_in_executor(None, agent_executor.invoke, {"input": query})
    output = response.get("output", "No output returned.")
    # If the response is too long for one message, split it into chunks.
    for chunk in [output[i:i+1900] for i in range(0, len(output), 1900)]:
        await ctx.send(chunk)

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    print("Welcome to the Airtable Agent. Type 'exit' to quit or use the Discord bot.")
    # Run the Discord bot
    bot.run(DISCORD_BOT_TOKEN)
