import httpx
from document_upload.app.models.document import store_document

async def ingest_from_api(data):
    token = None
    if data.api_connection_info.auth_type == "oauth2":
        async with httpx.AsyncClient() as client:
            response = await client.post(data.api_connection_info.token_url, data={
                "grant_type": "client_credentials",
                "client_id": data.api_connection_info.client_id,
                "client_secret": data.api_connection_info.client_secret
            })
            response.raise_for_status()
            token = response.json()["access_token"]

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient() as client:
        response = await client.get(data.api_connection_info.data_url, headers=headers)
        response.raise_for_status()
        document_text = response.text

    return await store_document(data, document_text)
