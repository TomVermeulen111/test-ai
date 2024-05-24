from typing import List
from langchain_core.document_loaders import BaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import requests
import os
import json
import re

class ComanLoader(BaseLoader):
    # Document loader for all content types in the Coman API.
    def __init__(self, schemeId: str, allowedSchemeFields: List[str], type: str) -> None:
        # Args:
        #     schemeId: The id of the scheme from which we load the content items.
        #     allowedSchemeFields: The fields that we want to include from the content items.
        #     type: The type of the content items.
        self.url = str(os.getenv("ORIS_COMAN_API_URL")) + "v1/odata/contentitem?$count=true&$orderby=DateCreated%20desc&$skip=0&$expand=Values,Scheme&$filter=(Scheme/Id%20eq%20" + schemeId + ")%20and%20(Status%20eq%20%27Active%27)"
        self.allowedSchemeFields = allowedSchemeFields
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
            categories = ""
            domains = ""
            language = ""
            links = []
            for value in contentItem["values"]:
                if (value["schemeField"]["name"] in self.allowedSchemeFields):
                    content += str(value["value"]) + " "
                if (value["schemeField"]["name"] == "IsPublic" or value["schemeField"]["name"] == "Article_public"):
                    isPublic = value["value"] == 'true' or False
                if (value["schemeField"]["name"] == "Date_publication"):
                    datePublication = value["value"] or contentItem["dateCreated"]
                if (value["schemeField"]["name"] == "Type" and value["value"]):
                    subAsJson = json.loads(value["value"])
                    subType = subAsJson["label"] or ''
                if (value["schemeField"]["name"] == "Subtype" and value["value"]):
                    # specifically for Actua type
                    subAsJson = json.loads(value["value"])
                    subType += ' - ' + str(subAsJson["label"]) or ''
                if (value["schemeField"]["name"] == "Domain" and value["value"]):
                    domainsAsJson = json.loads(value["value"])
                    domains = ", ".join(domain["label"] for domain in domainsAsJson)
                if (value["schemeField"]["name"] == "Language" and value["value"]):
                    # specifically for Actua type
                    languageAsJson = json.loads(value["value"])
                    language = languageAsJson["label"] or ''
                if (value["schemeField"]["name"] == "Category" and value["value"]):
                    categoriesAsJson = json.loads(value["value"])
                    categories = ", ".join(category["label"] for category in categoriesAsJson)
                # if ("file" in value["schemeField"]["name"].lower() and ("reference" not in value["schemeField"]["name"].lower()) and value["value"]):
                #     # get all files (not reference_files), load them using PyPDFLoader and add them to content
                #     filesAsJson = json.loads(value["value"])
                #     consecutiveDotsRemover = re.compile(r'\.{3,}')
                #     for file in filesAsJson:
                #         try:
                #             links.append(file["documentLink"])
                #             loader = PyPDFLoader(file["documentLink"])
                #             docs = loader.load()
                #             for doc in docs:
                #                 docContent = doc.page_content.replace('\n', ' ')
                #                 docContent = consecutiveDotsRemover.sub('', docContent)
                #                 content += docContent + " "
                #         except:
                #             print("Error loading file: " + file["documentLink"])

            document = Document(
                page_content=content,
                source = str(contentItem["id"]),
                metadata={
                    "source": str(contentItem["id"]),
                    "type": self.type,
                    "is_public": str(isPublic),
                    "date": datePublication,
                    "sub_type": subType,
                    "categories": categories,
                    "domains": domains,
                    "language": language,
                    # "links":  ", ".join(str(link) for link in links),
                }
            )
            items.append(document)
        return items
