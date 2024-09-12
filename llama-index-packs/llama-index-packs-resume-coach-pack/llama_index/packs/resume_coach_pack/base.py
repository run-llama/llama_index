from llama_index.core import Settings
from llama_index.core.llama_pack.base import BaseLlamaPack
from typing import List, Dict, Optional
from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core import Settings
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.schema import Document

from llama_index.readers.file import (
    DocxReader,
    HWPReader,
    PDFReader,
    EpubReader,
    FlatReader,
    HTMLTagReader,
    ImageReader,
    IPYNBReader,
    MarkdownReader,
    MboxReader,
    PptxReader,
    PandasCSVReader,
    VideoAudioReader,
    UnstructuredReader,
    XMLReader,
    CSVReader,
    RTFReader,
)

# Backward compatibility
try:
    from llama_index.core.llms.llm import LLM
except ImportError:
    from llama_index.core.llms.base import LLM


class RCFileReader:
    DEFAULT_READERS = {
        ".pdf": PDFReader,
        ".docx": DocxReader,
        ".hwp": HWPReader,
        ".epub": EpubReader,
        ".html": HTMLTagReader,
        ".ipynb": IPYNBReader,
        ".md": MarkdownReader,
        ".mbox": MboxReader,
        ".pptx": PptxReader,
        ".csv": PandasCSVReader,
        ".xml": XMLReader,
        ".rtf": RTFReader,
        ".txt": FlatReader,
        ".csv": CSVReader,
        ".jpg": ImageReader,
        ".jpeg": ImageReader,
        ".png": ImageReader,
        ".gif": ImageReader,
        ".bmp": ImageReader,
        ".tiff": ImageReader,
        ".mp3": VideoAudioReader,
        ".wav": VideoAudioReader,
        ".ogg": VideoAudioReader,
        ".flac": VideoAudioReader,
        ".aac": VideoAudioReader,
        ".wma": VideoAudioReader,
        ".m4a": VideoAudioReader,
    }

    def __init__(self, file_path: Path, reader=None) -> None:
        """Initialize file reader with either the default reader or a custom reader

        Args:
            file_path (Path): _description_
            reader (_type_, optional): _description_. Defaults to None.
        """
        self.file_path = file_path

        if reader is None:
            file_type = self.get_file_type()
            if file_type not in RCFileReader.DEFAULT_READERS:
                self.reader = SimpleDirectoryReader(
                    input_files=[self.file_path],
                    file_extractor={"": UnstructuredReader()},
                )
            else:
                self.reader = SimpleDirectoryReader(
                    input_files=[self.file_path],
                    file_extractor={
                        file_type: RCFileReader.DEFAULT_READERS[file_type]()
                    },
                )
        else:
            self.reader = reader

    def read(self) -> List[Document]:
        """Get file content

        Returns:
            List[Document]: _description_
        """
        if type(self.reader) != SimpleDirectoryReader:
            file_type = self.get_file_type()
            return SimpleDirectoryReader(
                input_files=[self.file_path], file_extractor={file_type: self.parser}
            ).load_data()
        else:
            return self.reader.load_data()

    def get_file_type(self) -> str:
        """Get file type

        Returns:
            _type_: _description_
        """
        return self.file_path.suffix


class ResumeCoachPack(BaseLlamaPack):
    def __init__(
        self,
        job_description_path: str,
        resume_path: str,
        embed_model: BaseEmbedding,
        llm: LLM,
        reader=None,
        queries: Optional[Dict] = None,
    ) -> None:
        """Create a new instance of the ResumeCoachPack

        Args:
            job_description_path (str): _description_
            resume_path (str): _description_
            embed_model (BaseEmbedding): _description_
            llm (LLM): _description_
            reader (_type_, optional): _description_. Defaults to None.
        """
        self.job_description_path = job_description_path
        self.resume_path = resume_path

        self.resume_reader = RCFileReader(
            file_path=Path(self.resume_path), reader=reader
        )
        self.job_description_reader = RCFileReader(
            file_path=Path(self.job_description_path), reader=reader
        )

        Settings.llm = llm
        Settings.embed_model = embed_model

        if queries is None:
            self.queries = {
                "match": "You are an expert recruiter and resume reviewer. You have to give a percentage of how much the resume matches the job criteria and description. Explain each criterion and give a percentage for each. At the end give an evaluation based on the job keywords that are found in the resume.",
                "improvement": "You are an expert coach. You have to give advice on how to improve in order to get a better matching score for the job description by using the criteria and keywords. Use only hard skills that can be improved and make a short plan for each.",
            }
        else:
            self.queries = queries

    def run(self) -> Dict:
        """Run the whole process of comparing the resume with the job description and return a dictionary with the results containing the match and improvement

        Returns:
            Dict: _description_
        """
        resume = self.resume_reader.read()
        job_description = self.job_description_reader.read()

        (self.criteria, self.keywords) = self.process_job_description(job_description)

        return self.compare(resume, job_description)

    def get_criteria(self, query_engine: RetrieverQueryEngine) -> str:
        """Get job criteria based on the job description

        Args:
            query_engine (RetrieverQueryEngine): _description_

        Returns:
            str: _description_
        """
        query = "Act as a recruting supervisor. Make a list of criterias from the job description separated by new lines. Don't say anything else other than the criteria."
        return query_engine.query(query).response

    def get_keywords(self, query_engine: RetrieverQueryEngine) -> str:
        """Get job keywords based on the job description

        Args:
            query_engine (RetrieverQueryEngine): _description_

        Returns:
            str: _description_
        """
        query = "Extract the keywords relevant to the job description from the document separated by commas."
        return query_engine.query(query).response

    def process_job_description(self, job_description: List[Document]) -> List:
        """Process the job description to get the criteria and keywords

        Args:
            job_description (List[Document]): _description_

        Returns:
            List: _description_
        """
        index_jd = VectorStoreIndex.from_documents(job_description)
        query_jd = index_jd.as_query_engine()

        return [self.get_criteria(query_jd).split("\n"), self.get_keywords(query_jd)]

    def compare(self, resume: List[Document], job_description: List[Document]) -> Dict:
        """Make the comparison between the resume and the job description using the criteria provided in the constructor

        Args:
            resume (List[Document]): _description_
            job_description (List[Document]): _description_

        Returns:
            Dict: _description_
        """
        index_compare = VectorStoreIndex.from_documents(resume)
        query_compare = index_compare.as_query_engine()

        end_of_query = f"\nCRITERIA: {''.join(self.criteria)}"
        end_of_query += f"\nJOB DESCRIPTION:"
        for jd in job_description:
            end_of_query += f"\n{jd}"
        end_of_query += f"\nJOB KEYWORDS: {self.keywords}"

        result = {}

        for key, query in self.queries.items():
            result[key] = query_compare.query(query + end_of_query).response

        return result

