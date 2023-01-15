"""Simple reader that ."""
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

from gpt_index.readers.base import BaseReader
from gpt_index.readers.schema.base import Document


def _docx_reader(input_file: Path, errors: str) -> str:
    """Extract text from Microsoft Word."""
    try:
        import docx2txt
    except ImportError:
        raise ValueError("docx2txt is required to read Microsoft Word files.")

    text = docx2txt.process(input_file)

    return text


def _pdf_reader(input_file: Path, errors: str) -> str:
    """Extract text from PDF."""
    try:
        import PyPDF2
    except ImportError:
        raise ValueError("PyPDF2 is required to read PDF files.")
    text_list = []
    with open(input_file, "rb") as file:
        # Create a PDF object
        pdf = PyPDF2.PdfReader(file)

        # Get the number of pages in the PDF document
        num_pages = len(pdf.pages)

        # Iterate over every page
        for page in range(num_pages):
            # Extract the text from the page
            page_text = pdf.pages[page].extract_text()
            text_list.append(page_text)
    text = "\n".join(text_list)

    return text


def _image_parser(input_file: Path, errors: str) -> str:
    """Extract text from images using DONUT."""
    try:
        import torch
    except ImportError:
        raise ValueError("install pytorch to use the model")
    try:
        from transformers import DonutProcessor, VisionEncoderDecoderModel
    except ImportError:
        raise ValueError("transformers is required for using DONUT model.")
    try:
        import sentencepiece  # noqa: F401
    except ImportError:
        raise ValueError("sentencepiece is required for using DONUT model.")
    try:
        from PIL import Image
    except ImportError:
        raise ValueError("PIL is required to read image files.")

    processor = DonutProcessor.from_pretrained(
        "naver-clova-ix/donut-base-finetuned-cord-v2"
    )
    model = VisionEncoderDecoderModel.from_pretrained(
        "naver-clova-ix/donut-base-finetuned-cord-v2"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # load document image
    image = Image.open(input_file)

    # prepare decoder inputs
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids

    pixel_values = processor(image, return_tensors="pt").pixel_values

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(
        processor.tokenizer.pad_token, ""
    )
    # remove first task start token
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

    return sequence


DEFAULT_FILE_EXTRACTOR: Dict[str, Callable[[Path, str], str]] = {
    ".pdf": _pdf_reader,
    ".docx": _docx_reader,
    ".jpg": _image_parser,
    ".png": _image_parser,
    ".jpeg": _image_parser,
}


class SimpleDirectoryReader(BaseReader):
    """Simple directory reader.

    Can read files into separate documents, or concatenates
    files into one document text.

    Args:
        input_dir (str): Path to the directory.
        exclude_hidden (bool): Whether to exclude hidden files (dotfiles).
        errors (str): how encoding and decoding errors are to be handled,
              see https://docs.python.org/3/library/functions.html#open
        recursive (bool): Whether to recursively search in subdirectories.
            False by default.
        required_exts (Optional[List[str]]): List of required extensions.
            Default is None.
        file_extractor (Optional[Dict[str, Callable]]): A mapping of file
            extension to a function that specifies how to convert that file
            to text. See DEFAULT_FILE_EXTRACTOR.
        num_files_limit (Optional[int]): Maximum number of files to read.
            Default is None.
        file_metadata (Optional[Callable[str, Dict]]): A function that takes
            in a filename and returns a Dict of metadata for the Document.
            Default is None.
    """

    def __init__(
        self,
        input_dir: str,
        exclude_hidden: bool = True,
        errors: str = "ignore",
        recursive: bool = False,
        required_exts: Optional[List[str]] = None,
        file_extractor: Optional[Dict[str, Callable]] = None,
        num_files_limit: Optional[int] = None,
        file_metadata: Optional[Callable[[str], Dict]] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize with parameters."""
        super().__init__(verbose=verbose)
        self.input_dir = Path(input_dir)
        self.errors = errors

        self.recursive = recursive
        self.exclude_hidden = exclude_hidden
        self.required_exts = required_exts
        self.num_files_limit = num_files_limit

        self.input_files = self._add_files(self.input_dir)
        self.file_extractor = file_extractor or DEFAULT_FILE_EXTRACTOR
        self.file_metadata = file_metadata

    def _add_files(self, input_dir: Path) -> List[Path]:
        """Add files."""
        input_files = sorted(input_dir.iterdir())
        new_input_files = []
        dirs_to_explore = []
        for input_file in input_files:
            if input_file.is_dir():
                if self.recursive:
                    dirs_to_explore.append(input_file)
            elif self.exclude_hidden and input_file.name.startswith("."):
                continue
            elif (
                self.required_exts is not None
                and input_file.suffix not in self.required_exts
            ):
                continue
            else:
                new_input_files.append(input_file)

        for dir_to_explore in dirs_to_explore:
            sub_input_files = self._add_files(dir_to_explore)
            new_input_files.extend(sub_input_files)

        if self.num_files_limit is not None and self.num_files_limit > 0:
            new_input_files = new_input_files[0 : self.num_files_limit]

        # print total number of files added
        if self.verbose:
            print(
                f"> [SimpleDirectoryReader] Total files added: {len(new_input_files)}"
            )

        return new_input_files

    def load_data(self, concatenate: bool = False) -> List[Document]:
        """Load data from the input directory.

        Args:
            concatenate (bool): whether to concatenate all files into one document.
                If set to True, file metadata is ignored.
                False by default.

        Returns:
            List[Document]: A list of documents.

        """
        data = ""
        data_list = []
        metadata_list = []
        for input_file in self.input_files:
            if input_file.suffix in self.file_extractor:
                data = self.file_extractor[input_file.suffix](input_file, self.errors)
            else:
                # do standard read
                with open(input_file, "r", errors=self.errors) as f:
                    data = f.read()
            data_list.append(data)
            if self.file_metadata is not None:
                metadata_list.append(self.file_metadata(str(input_file)))

        if concatenate:
            return [Document("\n".join(data_list))]
        elif self.file_metadata is not None:
            return [Document(d, extra_info=m) for d, m in zip(data_list, metadata_list)]
        else:
            return [Document(d) for d in data_list]
