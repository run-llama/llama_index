from markitdown import MarkItDown
from llama_index.core.bridge.pydantic import BaseModel, model_validator
import os
from pathlib import Path
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from typing import Tuple, Optional, Union, List
from typing_extensions import Self


def is_empty(list_it: list) -> Tuple[bool, Optional[list]]:
    if len(list_it) == 0:
        return True, None
    return False, list_it


class ValidFilePath(BaseModel):
    file_path: Union[str, Path, List[str], List[Path]]

    @model_validator(mode="after")
    def validate_file_path(self) -> Self:
        if isinstance(self.file_path, str):
            if not Path(self.file_path).is_dir():
                if not Path(self.file_path).is_file():
                    raise ValueError("File or directory path does not exist")
                else:
                    dir_files = [self.file_path]
            else:
                dir_files = []
                for root, _, files in os.walk(self.file_path):
                    for el in files:
                        dir_files.append(os.path.join(root, el))
            self.file_path = dir_files
        elif isinstance(self.file_path, Path):
            if not self.file_path.is_dir():
                if not self.file_path.is_file():
                    raise ValueError("File or directory path does not exist")
                else:
                    dir_files = [self.file_path]
            else:
                dir_files = []
                for root, _, files in os.walk(self.file_path):
                    for el in files:
                        dir_files.append(os.path.join(root, el))
            self.file_path = dir_files
        empty, fls = is_empty(self.file_path)
        if empty:
            raise ValueError("There is no file to parse!")
        else:
            files = []
            if isinstance(fls[0], str):
                for fl in fls:
                    if Path(fl).is_file() and os.path.splitext(fl)[1] in [
                        ".docx",
                        ".html",
                        ".xml",
                        ".csv",
                        ".pdf",
                        ".pptx",
                        ".xlsx",
                        ".json",
                        ".zip",
                        ".txt",
                        "",
                        ".md",
                    ]:
                        files.append(fl)
            else:
                for fl in fls:
                    if fl.is_file() and os.path.splitext(fl)[1] in [
                        ".docx",
                        ".html",
                        ".xml",
                        ".csv",
                        ".pdf",
                        ".pptx",
                        ".xlsx",
                        ".json",
                        ".zip",
                        ".txt",
                        "",
                        ".md",
                    ]:
                        files.append(fl.__str__())
            self.file_path = files

        return self


class MarkItDownReader(BaseReader):
    """
    MarkItDownReader is a document reader that utilizes the MarkItDown parser to convert files or collections of files into Document objects.

    Methods
    -------
    load_data(file_path: str | Path | Iterable[str] | Iterable[Path]) -> List[Document]
        Loads and parses a directory (if `file_path` is `str` or `Path`) or a list of files specified by `file_path` using the MarkItDown parser.
        Returns a list of Document objects, each containing the text content and metadata such as file path, file type, and content length.

    """

    _reader: MarkItDown = MarkItDown()

    @classmethod
    def class_name(cls) -> str:
        """Get the name identifier of the class."""
        return "MarkItDownReader"

    def load_data(
        self,
        file_path: Union[str, Path, List[str], List[Path]],
        **kwargs,
    ) -> List[Document]:
        docs: List[Document] = []
        fl_pt = ValidFilePath(file_path=file_path)
        fs = fl_pt.file_path

        for f in fs:
            res = self._reader.convert(f)
            docs.append(
                Document(
                    text=res.text_content,
                    metadata={
                        "file_path": f.__str__(),
                        "file_type": os.path.splitext(f)[1],
                        "content_length": len(res.text_content),
                    },
                )
            )

        return docs
