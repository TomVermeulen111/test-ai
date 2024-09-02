from typing import (
    List,
)
from langchain_core.documents import Document
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFLoader():
    # Document loader for pdf documents.
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
        loader = PyPDFLoader(self.file_path, extract_images=False)
        docs = loader.load()
        consecutiveDotsRemover = re.compile(r'\.{3,}')

        # standard PyPDFLoader chunks the pdf per page

        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=700,
        #     chunk_overlap=0,
        #     length_function=len
        # )
        # splitted = text_splitter.split_documents(docs)
        for idx, split in enumerate(docs):
            content = split.page_content.replace('\n', ' ')
            content = consecutiveDotsRemover.sub('', content)
            split.page_content = content
            # convert to string to avoid errors when uploading to azure search vector store
            split.metadata["page"] = str(split.metadata["page"])
            split.metadata["chunk_number"] = str(idx)
            split.metadata["document_id"] = str(self.document_id)
            split.metadata["document_name"] = str(self.document_name)
            split.metadata["document_link"] = str(self.file_path)
            split.metadata["document_type"] = str(self.document_type)
        return docs
    
    #TODO: Implement pdf's with images in them (OCR <=> Vision model (4o))