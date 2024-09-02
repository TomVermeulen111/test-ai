from typing import List
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from loaders import PDFLoader
import requests
import os
import json

class ComanLoader(BaseLoader):
    # Document loader for all content types in the Coman API.
    def __init__(self, schemeId: str, allowedSchemeFields: List[str], imageSchemeField: str, type: str) -> None:
        """Args:
            schemeId: The id of the scheme from which we load the content items.
            allowedSchemeFields: The fields that we want to include from the content items.
            type: The type of the content items."""
        self.url = str(os.getenv("ORIS_COMAN_API_URL")) + "v1/odata/contentitem?$count=true&$orderby=DateCreated%20desc&$skip=0&$expand=Values,Scheme&$filter=(Scheme/Id%20eq%20" + schemeId + ")%20and%20(Status%20eq%20%27Active%27)"
        self.allowedSchemeFields = allowedSchemeFields
        self.imageSchemeField = imageSchemeField
        self.type = type

    def lazy_load(self) -> List[Document]:  # <-- Does not take any arguments
        headers = {
            "Authorization": "apikey " + str(os.getenv("ORIS_COMAN_API_KEY")),
        }
        comanRequest = requests.get(self.url, headers=headers)
        comanRequestJson = comanRequest.json()
        contentItems = comanRequestJson["value"]
        items = []
        
        for contentItem in contentItems:
            content = ""
            isPublic = False
            datePublication = ""
            subType = ""
            categories = []
            domains = []
            language = ""
            title = ""
            attachments = []
            image = ""
            link = ""
            for value in contentItem["values"]:
                if (value["schemeField"]["name"] in self.allowedSchemeFields):
                    content += str(value["value"]) + " "
                if (value["schemeField"]["name"] == self.imageSchemeField):
                    if (value["value"]):
                        imageAsJson = json.loads(value["value"])
                        if (imageAsJson):
                            for file in imageAsJson:
                                image = file["documentLink"]
                if (value["schemeField"]["name"] == "IsPublic" or value["schemeField"]["name"] == "Article_public"):
                    isPublic = value["value"] == 'true' or False
                if (value["schemeField"]["name"] == "Date_publication"):
                    datePublication = value["value"]
                if (value["schemeField"]["name"] == "Type" and value["value"]):
                    subAsJson = json.loads(value["value"])
                    subType = subAsJson["label"] or ''
                if (value["schemeField"]["name"] == "Subtype" and value["value"]):
                    # specifically for Actua type
                    subAsJson = json.loads(value["value"])
                    subType += ' - ' + str(subAsJson["label"]) or ''
                if (value["schemeField"]["name"] == "Domain" and value["value"]):
                    domainsAsJson = json.loads(value["value"])
                    domains = [domain["label"] for domain in domainsAsJson]
                if (value["schemeField"]["name"] == "Language" and value["value"]):
                    # specifically for Actua type
                    languageAsJson = json.loads(value["value"])
                    language = languageAsJson["label"] or ''
                if (value["schemeField"]["name"] == "Category" and value["value"]):
                    categoriesAsJson = json.loads(value["value"])
                    categories = [category["label"] for category in categoriesAsJson]
                if (value["schemeField"]["name"] == "Title" or value["schemeField"]["name"] == "Name" ):
                    title = str(value["value"])
                if (value["schemeField"]["name"] == "Url"):
                    link = str(value["value"])
                if ("file" in value["schemeField"]["name"].lower() and ("reference" not in value["schemeField"]["name"].lower()) and value["value"]):
                    # get all files (not reference_files), load them using PyPDFLoader and add them to content
                    filesAsJson = json.loads(value["value"])
                    for file in filesAsJson:
                        if '.pdf' in file["name"]:
                            try:
                                loader = PDFLoader.PDFLoader(file["documentLink"], file["documentId"], file["name"], value["schemeField"]["name"])
                                attachmentDocs = loader.lazy_load()
                                attachments += attachmentDocs
                            except Exception as e:
                                print(e)
                                print("Error loading file: " + file["documentLink"])

            # Add contentitem as a document
            document = Document(
                page_content=content,
                source = str(contentItem["id"]),
                metadata={
                    "source": str(contentItem["id"]),
                    "type": self.type,
                    "is_public": str(isPublic),
                    "date": datePublication or contentItem["dateCreated"],
                    "sub_type": subType,
                    "categories": categories,
                    "domains": domains,
                    "language": language,
                    "title": title,
                    "image": image,
                    "link": link
                }
            )
            items.append(document)

            # Add ALL attachments to documents array with extra contentitem metadata
            for attachment in attachments:
                attachment.metadata["source"] = str(contentItem["id"])
                attachment.metadata["type"] = self.type
                attachment.metadata["is_public"] = str(isPublic)
                attachment.metadata["date"] = datePublication or contentItem["dateCreated"]
                attachment.metadata["sub_type"] = subType
                attachment.metadata["categories"] = categories
                attachment.metadata["domains"] = domains
                attachment.metadata["language"] = language
                attachment.metadata["title"] = title
                attachment.metadata["image"] = image
                attachment.metadata["link"] = link
                items.append(attachment)

        return items
