from typing import List
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
import requests
import os
import json

class ComanCollaboratorLoader(BaseLoader):
    # Document loader for all content types in the Coman API.
    def __init__(self) -> None:
        """Args:
            schemeId: The id of the scheme from which we load the content items.
            allowedSchemeFields: The fields that we want to include from the content items.
            type: The type of the content items."""
        self.url = str(os.getenv("ORIS_COMAN_API_URL")) + "v1/odata/contentitem?$count=true&$orderby=DateCreated%20desc&$skip=0&$expand=Values,Scheme&$filter=(Scheme/Id%20eq%20dee5bbfb-681a-47ac-91da-4663eedb48e4)%20and%20(Status%20eq%20%27Active%27)"

    def lazy_load(self) -> List[Document]:  # <-- Does not take any arguments
        headers = {
            "Authorization": "apikey " + str(os.getenv("ORIS_COMAN_API_KEY")),
        }
        comanRequest = requests.get(self.url, headers=headers)
        comanRequestJson = comanRequest.json()
        contentItems = comanRequestJson["value"]
        items = []
        
        for contentItem in contentItems:
            name = ""
            image = ""
            values = contentItem["values"]
            cibDepartment = [x.get("value") for x in values if x.get("schemeField").get("name") == "CIB_department"]
            cibDepartmentLabel = None
            if (cibDepartment and cibDepartment[0]):
                cibDepartmentJson = json.loads(cibDepartment[0])
                cibDepartmentLabel = str(cibDepartmentJson["label"])
            
            content = str([x.get("value") for x in values if x.get("schemeField").get("name") == "Name"][0]) + " is " + str([x.get("value") for x in values if x.get("schemeField").get("name") == "Function_one"][0]).lower()
            if (cibDepartmentLabel):
                content += " bij " + str(cibDepartmentLabel).lower()
            content += '.'
            telephone = [x.get("value") for x in values if x.get("schemeField").get("name") == "Telephone"][0]
            email = [x.get("value") for x in values if x.get("schemeField").get("name") == "Email"][0]

            if (telephone is not None and telephone.strip() != '')  or (email is not None and email.strip() != ''):
                content += " Te bereiken"
                if telephone is not None and telephone.strip() != '':
                    content += " via telefoonnummer: " + str(telephone)
                    if (email):
                        content += " en"

                if email is not None and email.strip() != '':
                    content += " via email: " + str(email)

            for value in contentItem["values"]:
                if (value["schemeField"]["name"] == "Title" or value["schemeField"]["name"] == "Name"):
                    name = str(value["value"])
                if (value["schemeField"]["name"] == "File"):
                    image = str(value["value"])

            # Add contentitem as a document
            document = Document(
                page_content=content,
                source = str(contentItem["id"]),
                metadata={
                    "source": str(contentItem["id"]),
                    "type": 'Medewerkers/Personen',
                    "is_public": str(False),
                    "date": contentItem["dateCreated"],
                    "title": name,
                    "image": image,
                }
            )
            items.append(document)

        return items
