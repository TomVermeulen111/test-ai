from podcasts import SpeechToTextConverter
import os
from langchain_core.document_loaders import BaseLoader
from typing import (
    List,
)
from langchain_core.documents import Document

from podcasts.PodcastType import PodcastType
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PodcastLoader(BaseLoader):
    def __init__(self, real_estate_actual_folder_path: str, real_estate_talk_folder_path: str) -> None:
        self.real_estate_actual_folder_path = real_estate_actual_folder_path
        self.real_estate_talk_folder_path = real_estate_talk_folder_path

    def lazy_load(self) -> List[Document]:
        docs = []
        real_estate_talk_docs = self.TranscribePodcasts(self.real_estate_talk_folder_path, PodcastType.REAL_ESTATE_TALK)
        # real_estate_actual_docs = self.TranscribePodcasts(self.real_estate_actual_folder_path, PodcastType.REAL_ESTATE_ACTUAL)
        docs.extend(real_estate_talk_docs)
        # docs.extend(real_estate_actual_docs)        

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=100,
            length_function=len
        )
        splitted_docs = text_splitter.split_documents(docs)

        return splitted_docs

    def TranscribePodcasts(self, podcasts_folder_path: str, podcasts_type: PodcastType) -> List[Document]:
        wav_files = [f for f in os.listdir(podcasts_folder_path) if f.endswith('.wav')]

        items = []
        for file in wav_files:
            audio_file_path = os.path.join(podcasts_folder_path, file)
            transcription = SpeechToTextConverter.RecognizeTextFromAudioFile(audio_file_path)

            document = Document(
                page_content=transcription,
                source=audio_file_path,
                metadata={
                    "source": audio_file_path,
                    "type": podcasts_type.value,
                    "title": file.split('.')[0]
                }
            )
            items.append(document)

        return items