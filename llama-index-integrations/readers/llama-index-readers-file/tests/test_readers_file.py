from llama_index.core.readers.base import BaseReader
from llama_index.readers.file import (
    DocxReader,
    EpubReader,
    FlatReader,
    HWPReader,
    ImageCaptionReader,
    ImageReader,
    ImageVisionLLMReader,
    IPYNBReader,
    MarkdownReader,
    MboxReader,
    PandasCSVReader,
    PDFReader,
    PptxReader,
    VideoAudioReader,
    XMLReader,
    ImageTabularChartReader,
)


def test_classes():
    names_of_base_classes = [b.__name__ for b in DocxReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in HWPReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in PDFReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in EpubReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in FlatReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in ImageReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in ImageCaptionReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in ImageVisionLLMReader.__mro__]
    ###ALEX
    # assert BaseReader.__name__ in names_of_base_classes
    # assert hasattr(ImageVisionLLMReader, "__init__")
    # assert hasattr(ImageVisionLLMReader, "load_data")
    # assert hasattr(ImageVisionLLMReader, "_import_torch")
    # image_vision_llm_reader = ImageVisionLLMReader(
    #     parser_config={
    #         "processor": None,
    #         "model": None,
    #         "device": "cpu",
    #         "dtype": float,
    #     }
    # )
    # assert hasattr(image_vision_llm_reader, "_torch")
    # assert hasattr(image_vision_llm_reader, "_torch_imported")
    # image_vision_llm_reader._import_torch()
    # print(f'\nZZZZZZZZZZ\n{image_vision_llm_reader._torch}')
    # print(f'\nYYYYYYYYYY\n{image_vision_llm_reader._torch_imported}')
    # # b = image_vision_llm_reader.load_data(file=None)
    # # image_vision_llm_reader = ImageVisionLLMReader()
    # # result = image_vision_llm_reader.load_data(file="/Users/alexsherstinsky/Desktop/BookImages/BuildALargeLanguageModelFromScratch.jpg")
    # # print(result)
    # # print(str(type(result)))
    ###ALEX

    names_of_base_classes = [b.__name__ for b in IPYNBReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in MarkdownReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in MboxReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in PptxReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in PandasCSVReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in VideoAudioReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in XMLReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in ImageTabularChartReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes
