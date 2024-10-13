# This file is adapted from
# https://github.com/HabanaAI/Gaudi-tutorials/blob/main/PyTorch/localGPT_inference/gaudi_utils/embeddings.py
#
import json
import os
from collections import OrderedDict

import numpy as np
import torch
from InstructorEmbedding import INSTRUCTOR_Pooling, INSTRUCTOR_Transformer
from InstructorEmbedding.instructor import batch_to_device, import_from_string
from sentence_transformers import SentenceTransformer
from tqdm.autonotebook import trange

from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings


class GaudiSentenceTransformer(SentenceTransformer):
    """Child class that overrides the tokenize method from SentenceTransformer."""

    def __init__(self, model_name_or_path, embedding_input_size=-1, **kwargs) -> None:
        super().__init__(model_name_or_path, **kwargs)
        self.embedding_input_size = embedding_input_size

    def tokenize(self, texts):
        """Override tokenize method from SentenceTransformer."""
        return self._first_module().tokenizer(
            texts,
            max_length=self.max_seq_length
            if (
                self.embedding_input_size == -1
                or self.embedding_input_size > self.max_seq_length
            )
            else self.embedding_input_size,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )


class GaudiHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    """Child class that uses a GaudiSentenceTransformer client."""

    def __init__(self, embedding_input_size=-1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.client = GaudiSentenceTransformer(
            self.model_name,
            embedding_input_size=embedding_input_size,
            cache_folder=self.cache_folder,
            **self.model_kwargs,
        )


class GaudiINSTRUCTOR(GaudiSentenceTransformer):
    """INSTRUCTOR class for running on Gaudis. Code taken from instructor-embedding repo."""

    def __init__(self, model_name_or_path, embedding_input_size=-1, **kwargs) -> None:
        super().__init__(
            model_name_or_path, embedding_input_size=embedding_input_size, **kwargs
        )

    def smart_batching_collate(self, batch):
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text)

            labels.append(example.label)

        labels = torch.tensor(labels)

        sentence_features = []
        for idx in range(num_texts):
            assert isinstance(texts[idx][0], list)
            assert (
                len(texts[idx][0]) == 2
            ), "The input should have both instruction and input text"
            # if len(texts[idx][0])==3:
            # print('component 3')
            num = len(texts[idx])
            contexts = []
            concatenated_input_texts = []
            for local_idx in range(num):
                assert len(texts[idx][local_idx]) == 2
                contexts.append(texts[idx][local_idx][0])
                concatenated_input_texts.append("".join(texts[idx][local_idx]))
                assert isinstance(contexts[-1], str)
                assert isinstance(concatenated_input_texts[-1], str)
            tokenized = self.tokenize(concatenated_input_texts)
            context_tok = self.tokenize(contexts)
            tokenized["context_masks"] = torch.sum(context_tok["attention_mask"], dim=1)
            tokenized["context_masks"] = tokenized["context_masks"] - 1
            for my_idx in range(len(tokenized["context_masks"])):
                if tokenized["context_masks"][my_idx] <= 1:
                    tokenized["context_masks"][my_idx] = 0
                # text_types = [pair[-1] for pair in texts[idx]]
                # assert all([tid==1 for tid in text_types]) or all([tid==0 for tid in text_types])
                # tokenized['text_type'] = text_types[0]
            # elif len(texts[idx][0])==2:
            #     input_texts = [pair[0] for pair in texts[idx]]
            #     text_types = [pair[-1] for pair in texts[idx]]
            #     assert all([tid == 1 for tid in text_types]) or all([tid == 0 for tid in text_types])
            #     tokenized = self.tokenize(input_texts)
            #     tokenized['text_type'] = text_types[0]
            # else:
            #     raise ValueError('tokenization error')
            sentence_features.append(tokenized)

        return sentence_features, labels

    def _load_sbert_model(self, model_path):
        """Loads a full sentence-transformers model."""
        # Check if the config_sentence_transformers.json file exists (exists since v2 of the framework)
        config_sentence_transformers_json_path = os.path.join(
            model_path, "config_sentence_transformers.json"
        )
        if os.path.exists(config_sentence_transformers_json_path):
            with open(config_sentence_transformers_json_path) as fIn:
                self._model_config = json.load(fIn)

        # Check if a readme exists
        model_card_path = os.path.join(model_path, "README.md")
        if os.path.exists(model_card_path):
            try:
                with open(model_card_path, encoding="utf8") as fIn:
                    self._model_card_text = fIn.read()
            except Exception:
                pass

        # Load the modules of sentence transformer
        modules_json_path = os.path.join(model_path, "modules.json")
        with open(modules_json_path) as fIn:
            modules_config = json.load(fIn)

        modules = OrderedDict()
        for module_config in modules_config:
            if module_config["idx"] == 0:
                print("load INSTRUCTOR_Transformer")
                module_class = INSTRUCTOR_Transformer
            elif module_config["idx"] == 1:
                module_class = INSTRUCTOR_Pooling
            else:
                module_class = import_from_string(module_config["type"])
            module = module_class.load(os.path.join(model_path, module_config["path"]))
            modules[module_config["name"]] = module

        return modules

    def encode(
        self,
        sentences,
        batch_size: int = 32,
        show_progress_bar: bool = None,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str = None,
        normalize_embeddings: bool = False,
    ):
        """
        Computes sentence embeddings.

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings. Set to None, to get all output values
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = False

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != "sentence_embedding":
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(
            sentences, "__len__"
        ):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = self._target_device

        self.to(device)

        all_embeddings = []
        if isinstance(sentences[0], list):
            lengths = []
            for sen in sentences:
                lengths.append(-self._text_length(sen[1]))
            length_sorted_idx = np.argsort(lengths)
        else:
            length_sorted_idx = np.argsort(
                [-self._text_length(sen) for sen in sentences]
            )
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(
            0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar
        ):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            features = self.tokenize(sentences_batch)
            features = batch_to_device(features, device)

            with torch.no_grad():
                out_features = self.forward(features)

                if output_value == "token_embeddings":
                    embeddings = []
                    for token_emb, attention in zip(
                        out_features[output_value], out_features["attention_mask"]
                    ):
                        last_mask_id = len(attention) - 1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1

                        embeddings.append(token_emb[0 : last_mask_id + 1])
                elif output_value is None:  # Return all outputs
                    embeddings = []
                    for sent_idx in range(len(out_features["sentence_embedding"])):
                        row = {
                            name: out_features[name][sent_idx] for name in out_features
                        }
                        embeddings.append(row)
                else:  # Sentence embeddings
                    embeddings = out_features[output_value]
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(
                            embeddings, p=2, dim=1
                        )

                    # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                    if convert_to_numpy:
                        embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings


class GaudiHuggingFaceInstructEmbeddings(HuggingFaceInstructEmbeddings):
    """Child class that uses a GaudiINSTRUCTOR client."""

    def __init__(self, embedding_input_size=-1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.client = GaudiINSTRUCTOR(
            self.model_name,
            embedding_input_size=embedding_input_size,
            cache_folder=self.cache_folder,
            **self.model_kwargs,
        )
