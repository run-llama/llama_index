import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class PDFNougatOCR(BaseReader):
    def nougat_ocr(self, file_path: Path) -> str:
        cli_command = ["nougat", "--markdown", "pdf", str(file_path), "--out", "output"]

        try:
            result = subprocess.run(cli_command, capture_output=True, text=True)
            result.check_returncode()
            return result.stdout

        except subprocess.CalledProcessError as e:
            logging.error(
                f"Nougat OCR command failed with return code {e.returncode}: {e.stderr}"
            )
            raise RuntimeError("Nougat OCR command failed.") from e

    def load_data(
        self, file_path: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        try:
            # Ensure the 'output' folder exists or create it if not
            output_folder = Path("output")
            output_folder.mkdir(exist_ok=True)

            # Call the method to run the Nougat OCR command
            self.nougat_ocr(file_path)

            # Rest of your code for reading and processing the output
            file_path = Path(file_path)
            output_path = output_folder / f"{file_path.stem}.mmd"
            with output_path.open("r") as f:
                content = f.read()

            content = (
                content.replace(r"\(", "$")
                .replace(r"\)", "$")
                .replace(r"\[", "$$")
                .replace(r"\]", "$$")
            )

            # Need to chunk before creating Document

            return [Document(text=content)]

        except Exception as e:
            logging.error(f"An error occurred while processing the PDF: {e!s}")
