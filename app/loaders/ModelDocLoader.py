from typing import List
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from loaders import ModelDocPdfLoader
import requests
import os

class ModelDocLoader(BaseLoader):
    # Document loader for all content types in the Coman API.
    def __init__(self) -> None:
        """Args:
            schemeId: The id of the scheme from which we load the content items.
            allowedSchemeFields: The fields that we want to include from the content items.
            type: The type of the content items."""
        self.url = str(os.getenv("ORIS_MODELDOCS_API_URL"))

    def lazy_load(self) -> List[Document]:  # <-- Does not take any arguments
        # Reuse same api key
        headers = {
            "Authorization": "apikey " + str(os.getenv("ORIS_COMAN_API_KEY")),
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        listUrl = self.url + "api/Publication/List"
        listJson = {
            "pageSize": 1500,
            "pageIndex": 1,
            "publicationChannel": "CIBweb",
            "accessRights": [
            ],
            "reseller": "CIB",
            "categories": [
            ],
            "languages": [
            ],
            "locationRealEstates": [
            ]
        }

        # Get initial full list of documents
        modelDocsRequest = requests.post(listUrl, json=listJson, headers=headers)
        modelDocsRequestJson = modelDocsRequest.json()
        modelDocs = modelDocsRequestJson["items"]
        items = []
        
        for modelDoc in modelDocs :
            id = modelDoc["id"]
            attachments = []
            headers = {
                "Authorization": "apikey " + str(os.getenv("ORIS_COMAN_API_KEY")),
                "X-Immoconnect-Sessionid": str(os.getenv("ORIS_SESSION_ID"))
            }
            getShareLinkUrl = self.url + f"api/Publication/GetRevisionDocumentShareLink?documentId={id}&fileType=Pdf"
            modelDocShareLink = requests.get(getShareLinkUrl, headers=headers)
            modelDocShareLinkResult = modelDocShareLink.text

            try:
                loader = ModelDocPdfLoader.ModelDocPdfLoader(modelDocShareLinkResult, None, modelDoc["name"], modelDoc["category"])
                attachmentDocs = loader.lazy_load()
                attachments += attachmentDocs
            except Exception as e:
                print(e)
                print("Error loading file: " + modelDocShareLinkResult + " with id: " + str(id))

            # Add ALL attachments to documents array with extra contentitem metadata
            for attachment in attachments:
                attachment.page_content = str(modelDoc["code"]) + " - " + str(modelDoc["name"]) + ' - ' + str(modelDoc["locationRealEstate"]) + ' - '  + attachment.page_content
                attachment.metadata["source"] = str(id)
                attachment.metadata["type"] = 'Modeldocument'
                attachment.metadata["is_public"] = 'False'
                attachment.metadata["date"] = str(modelDoc["lastPublicationVersionDate"])
                attachment.metadata["categories"] = str(modelDoc["cibCategories"])
                attachment.metadata["domains"] = str(modelDoc["cibDomains"])
                attachment.metadata["language"] = str(modelDoc["language"])
                attachment.metadata["title"] = str(modelDoc["name"])
                attachment.metadata["lastPublicationDocumentId"] = str(modelDoc["lastPublicationDocumentId"])
                attachment.metadata["lastPublicationVersionComment"] = str(modelDoc["lastPublicationVersionComment"])
                attachment.metadata["lastPublicationVersion"] = str(modelDoc["lastPublicationVersion"])
                attachment.metadata["lastPublicationVersion"] = str(modelDoc["lastPublicationVersion"])
                attachment.metadata["locationRealEstate"] = str(modelDoc["locationRealEstate"])
                attachment.metadata["code"] = str(modelDoc["code"])
                items.append(attachment)
                # print(attachment)

        return items
