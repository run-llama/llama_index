"""
Reader that uses a Github Repo.

Repo taken from: https://github.com/mpoon/gpt-repository-loader

License attached:

MIT License

Copyright (c) 2023 mpoon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

#!/usr/bin/env python3

import fnmatch
import os
from typing import List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


def get_ignore_list(ignore_file_path) -> List[str]:
    ignore_list = []
    with open(ignore_file_path) as ignore_file:
        for line in ignore_file:
            ignore_list.append(line.strip())
    return ignore_list


def should_ignore(file_path, ignore_list) -> bool:
    for pattern in ignore_list:
        if fnmatch.fnmatch(file_path, pattern):
            return True
    return False


def process_repository(
    repo_path,
    ignore_list,
    concatenate: bool = False,
    extensions: Optional[List[str]] = None,
    encoding: Optional[str] = "utf-8",
) -> List[str]:
    """Process repository."""
    result_texts = []
    result_text = ""
    for root, _, files in os.walk(repo_path):
        for file in files:
            file_path = os.path.join(root, file)
            relative_file_path = os.path.relpath(file_path, repo_path)

            _, file_ext = os.path.splitext(file_path)
            is_correct_extension = extensions is None or file_ext in extensions

            if (
                not should_ignore(relative_file_path, ignore_list)
                and is_correct_extension
            ):
                with open(file_path, errors="ignore", encoding=encoding) as file:
                    contents = file.read()
                result_text += "-" * 4 + "\n"
                result_text += f"{relative_file_path}\n"
                result_text += f"{contents}\n"
                if not concatenate:
                    result_texts.append(result_text)
                    result_text = ""

    if concatenate:
        result_texts.append(result_text)

    return result_texts


class GPTRepoReader(BaseReader):
    """
    GPTRepoReader.

    Reads a github repo in a prompt-friendly format.

    """

    def __init__(self, concatenate: bool = False) -> None:
        """Initialize."""
        self.concatenate = concatenate

    def load_data(
        self,
        repo_path: str,
        preamble_str: Optional[str] = None,
        extensions: Optional[List[str]] = None,
        encoding: Optional[str] = "utf-8",
    ) -> List[Document]:
        """
        Load data from the input directory.

        Args:
            pages (List[str]): List of pages to read.

        """
        ignore_file_path = os.path.join(repo_path, ".gptignore")

        if os.path.exists(ignore_file_path):
            ignore_list = get_ignore_list(ignore_file_path)
        else:
            ignore_list = []

        output_text = ""
        if preamble_str:
            output_text += f"{preamble_str}\n"
        elif self.concatenate:
            output_text += (
                "The following text is a Git repository with code. "
                "The structure of the text are sections that begin with ----, "
                "followed by a single line containing the file path and file "
                "name, followed by a variable amount of lines containing the "
                "file contents. The text representing the Git repository ends "
                "when the symbols --END-- are encountered. Any further text beyond "
                "--END-- are meant to be interpreted as instructions using the "
                "aforementioned Git repository as context.\n"
            )
        else:
            # self.concatenate is False
            output_text += (
                "The following text is a file in a Git repository. "
                "The structure of the text are sections that begin with ----, "
                "followed by a single line containing the file path and file "
                "name, followed by a variable amount of lines containing the "
                "file contents. The text representing the file ends "
                "when the symbols --END-- are encountered. Any further text beyond "
                "--END-- are meant to be interpreted as instructions using the "
                "aforementioned file as context.\n"
            )
        text_list = process_repository(
            repo_path,
            ignore_list,
            concatenate=self.concatenate,
            extensions=extensions,
            encoding=encoding,
        )
        docs = []
        for text in text_list:
            doc_text = output_text + text + "\n--END--\n"
            docs.append(Document(text=doc_text))

        return docs
