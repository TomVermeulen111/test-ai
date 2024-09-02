from typing import (
    List,
)
from langchain_core.documents import Document
import re
import requests

import pymupdf

class ModelDocPdfLoader():
    """ Document loader for model documents. Splits on clausules"""
    def __init__(
        self, 
        file_path: str,
        document_id: str,
        document_name: str,
        document_type: str
    ) -> None:
        """Args:
            file_path: The filepath of the PDF (web url) we're trying to download."""
        self.file_path = file_path
        self.document_id = document_id
        self.document_name = document_name
        self.document_type = document_type

    def lazy_load(self) -> List[Document]:  # <-- Does not take any arguments
        docs = self.load_model_doc()

        for idx, split in enumerate(docs):
            # convert to string to avoid errors when uploading to azure search vector store
            split.metadata["chunk_number"] = str(idx)
            split.metadata["document_id"] = str(self.document_id)
            split.metadata["document_name"] = str(self.document_name)
            split.metadata["document_link"] = str(self.file_path)
            split.metadata["document_type"] = str(self.document_type)
        return docs
    
    def load_model_doc(self):
        """
        Extract text from a model doc and split it per clausule
        
        Args:
            pdf_path (str): Path to the model doc PDF document.
            
        Returns:
            List[Document]: A list of documents, each document contains a clausule (or a complete article in case it has no clausules).
        """

        res = requests.get(self.file_path)
        doc = pymupdf.open(stream=res.content)
        text = ""
        for page in doc:
            rect = page.rect
            height = 50
            # clip is used to cut off the bottom to remove the footer (containing page number and model doc code)
            clip = pymupdf.Rect(0, height, rect.width, rect.height-height)
            text += page.get_textbox(clip)

        clausules = self.split_in_clausules(text)

        documents = []
        for clausule in clausules:
            doc = Document(
                page_content=clausule,
            )
            documents.append(doc)

        return documents
    

    def split_in_clausules(self, text):
        """
        Split the text in clausules.
        If an article has no separate clausules, the clausule is the article itself.
        
        Args:
            text (str): The text of the model doc.
            
        Returns:
            List[str]: A list of clausules.
        """
        articleHeaderRegex = re.compile(r"(ARTIKEL|ARTICLE) \d+\.")
        clausuleHeaderRegex = re.compile(r"^\d+\.\d+\s.*")
        clausules = []
        current_article_title =''
        current_clausule = ''

        for line in text.splitlines():
            if re.match(articleHeaderRegex, line):
                clausules.append(current_clausule)
                current_article_title = line
                current_clausule = ''
            if re.match(clausuleHeaderRegex, line):
                clausules.append(current_clausule)
                current_clausule = current_article_title + '\n'

            current_clausule += line

        clausules.append(current_clausule)

        return clausules