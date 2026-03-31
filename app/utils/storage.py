import os
import logging
from azure.storage.blob import BlobServiceClient
from app.config import settings

logger = logging.getLogger("viz-agent")

class StorageProvider:
    def __init__(self):
        self.connection_string = settings.AZURE_STORAGE_CONNECTION_STRING
        self.container_name = settings.AZURE_STORAGE_CONTAINER_NAME
        self.use_azure = settings.STORAGE_TYPE == "azure_blob" and bool(self.connection_string)
        
        if self.use_azure:
            try:
                self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
                self.container_client = self.blob_service_client.get_container_client(self.container_name)
                logger.info("Azure Blob Storage initialized for Viz Agent.")
            except Exception as e:
                logger.error(f"Failed to initialize Azure Blob Storage: {e}")
                self.use_azure = False

    def save_chart(self, content: bytes, filename: str) -> str:
        """Saves chart and returns its path or URL."""
        if self.use_azure:
            blob_client = self.container_client.get_blob_client(filename)
            blob_client.upload_blob(content, overwrite=True)
            return blob_client.url
        else:
            # Fallback to local
            os.makedirs(settings.CHART_OUTPUT_PATH, exist_ok=True)
            file_path = os.path.join(settings.CHART_OUTPUT_PATH, filename)
            with open(file_path, "wb") as f:
                f.write(content)
            return file_path

storage = StorageProvider()
