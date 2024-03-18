import regex
import string
import json
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from transformers import DPRReader, DPRReaderTokenizer
import torch
from typing import Any
from sentence_transformers import CrossEncoder
import time


def _normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _match_or_not(prediction, ground_truth):
    norm_predict = _normalize_answer(prediction)
    norm_answer = _normalize_answer(ground_truth)
    return norm_answer in norm_predict


def _have_seen_or_not(model_cross_encoder, query_item, query_seen_list, query_type):
    if "Unsolved" in query_type:
        return False
    for query_seen in query_seen_list:
        with torch.no_grad():
            if model_cross_encoder.predict([(query_seen, query_item)]) > 0.5:
                return True
    return False


class SearChainPack(BaseLlamaPack):
    """Simple short form SearChain pack."""

    def __init__(
        self,
        data_path: str,
        dprtokenizer_path: str = "/dpr_reader_multi",
        dprmodel_path: str = "/dpr_reader_multi",
        crossencoder_name_or_path: str = "/Quora_cross_encoder",
        device: str = "cuda",
        **kwargs: Any,
    ) -> None:
        """Init params."""
        self.device = device
        self.crossencoder = CrossEncoder(crossencoder_name_or_path, device=self.device)
        self.documents = SimpleDirectoryReader(data_path).load_data()
        self.index = VectorStoreIndex.from_documents(self.documents)
        self.query_engine = self.index.as_query_engine()
        self.llm = OpenAI()

        self.dprtokenizer = DPRReaderTokenizer.from_pretrained(dprtokenizer_path)
        self.dprmodel = DPRReader.from_pretrained(dprmodel_path)
        self.dprmodel.eval()
        self.dprmodel.to(self.device)

    def _get_answer(self, query, texts, title):
        print("texts:" + texts)
        encoded_inputs = self.dprtokenizer(
            questions=[query],
            titles=[title],
            texts=[texts],
            return_tensors="pt",
            max_length=510,
        )
        outputs = self.dprmodel(**encoded_inputs.to(self.device))
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        relevance_logits = outputs.relevance_logits

        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()
        predict_answer_tokens = encoded_inputs.input_ids[
            0, answer_start_index : answer_end_index + 1
        ]
        answer = self.dprtokenizer.decode(predict_answer_tokens)
        return answer, relevance_logits

    def _ir(self, query, query_seen_list):
        flag_contibue_label = False
        query_list = query.split("\n")
        message = ""
        for idx in range(len(query_list)):
            query_item = query_list[idx]
            if "Query" in query_item and "]:" in query_item:
                temp = query_item.split("]")
                if len(temp) < 2:
                    continue
                query_type = temp[0]
                query_item = temp[1]
                if ":" in query_item:
                    query_item = query_item[1:]
                if not _have_seen_or_not(
                    self.crossencoder, query_item, query_seen_list, query_type
                ):
                    now_reference = {}
                    query_seen_list.append(query_item)
                    response = str(self.query_engine.query(query_item))

                    answer, relevance_score = self._get_answer(
                        query=query_item, texts="", title=response
                    )
                    now_reference["query"] = query_item
                    now_reference["answer"] = answer
                    now_reference["reference"] = response
                    now_reference["ref_score"] = relevance_score
                    now_reference["idx"] = response

                    if "Unsolved" in query_type:
                        message = "[Unsolved Query]:{}<SEP>[Answer]:{}<SEP>[Reference]:{}<SEP>".format(
                            query_item, answer, response
                        )
                        flag_contibue_label = True
                        break
                    elif relevance_score > 1.5:
                        answer_start_idx = idx + 1
                        predict_answer = ""
                        while answer_start_idx < len(query_list):
                            if "Answer" in query_list[answer_start_idx]:
                                predict_answer = query_list[answer_start_idx]
                                break
                            answer_start_idx += 1
                        match_label = _match_or_not(
                            prediction=predict_answer, ground_truth=answer
                        )
                        if match_label:
                            continue
                        else:
                            message = "[Query]:{}<SEP>[Answer]:{}<SEP>[Reference]:{}<SEP>".format(
                                query_item, answer, response
                            )
                            flag_contibue_label = True
                            break
        return message, flag_contibue_label, query_seen_list

    def _extract(self, message_keys_list):
        text = message_keys_list
        idx = len(text)
        while idx > 0:
            idx = idx - 1
            item = text[idx]
            if item.role == "assistant" and "Final Content" in item.content:
                list_item = item.content.split("\n")
                for sp in list_item:
                    if "Final Content" in sp:
                        return item.content
        return "Sorry, I still cannot solve this question!"

    def execute(self, data_path, start_idx):
        data = open(data_path)
        for k, example in enumerate(data):
            if k < start_idx:
                continue
            example = json.loads(example)
            q = example["question"]
            round_count = 0
            message_keys_list = [
                ChatMessage(
                    role="user",
                    content="""Construct a global reasoning chain for this complex [Question] : " {} " You should generate a query to the search engine based on what you already know at each step of the reasoning chain, starting with [Query]. If you know the answer for [Query], generate it starting with [Answer]. You can try to generate the final answer for the [Question] by referring to the [Query]-[Answer] pairs, starting with [Final Content]. If you don't know the answer, generate a query to search engine based on what you already know and do not know, starting with [Unsolved Query].
                        For example:
                        [Question]: "Where do greyhound buses that are in the birthplace of Spirit If...'s performer leave from? "
                        [Query 1]: Who is the performer of Spirit If... ?
                        If you don't know the answer:
                        [Unsolved Query]: Who is the performer of Spirit If... ?
                        If you know the answer:
                        [Answer 1]: The performer of Spirit If... is Kevin Drew.
                        [Query 2]: Where was Kevin Drew born?
                        If you don't know the answer:
                        [Unsolved Query]: Where was Kevin Drew born?
                        If you know the answer:
                        [Answer 2]: Toronto.
                        [Query 3]: Where do greyhound buses in Toronto leave from?
                        If you don't know the answer:
                        [Unsolved Query]: Where do greyhound buses in Toronto leave from?
                        If you know the answer:
                        [Answer 3]: Toronto Coach Terminal.
                        [Final Content]: The performer of Spirit If... is Kevin Drew [1]. Kevin Drew was born in Toronto [2]. Greyhound buses in Toronto leave from Toronto Coach Terminal [3]. So the final answer is Toronto Coach Terminal.

                        [Question]:"Which magazine was started first Arthur’s Magazine or First for Women?"
                        [Query 1]: When was Arthur’s Magazine started?
                        [Answer 1]: 1844.
                        [Query 2]: When was First for Women started?
                        [Answer 2]: 1989
                        [Final Content]: Arthur’s Magazine started in 1844 [1]. First for Women started in 1989 [2]. So Arthur’s Magazine was started first. So the answer is Arthur’s Magazi
                        [Question]: {}""".format(
                        q, q
                    ),
                )
            ]
            feedback_answer = "continue"
            predict_answer = ""
            query_seen_list = []
            while round_count < 5 and feedback_answer != "end":
                time.sleep(0.5)
                rsp = self.llm.chat(message_keys_list)
                round_count += 1
                input_str = str(rsp.message.content)
                message_keys_list.append(
                    ChatMessage(role="assistant", content=input_str)
                )
                predict_answer += input_str

                message, flag_contibue_label, query_seen_list = self._ir(
                    input_str, query_seen_list
                )
                if flag_contibue_label:
                    feedback = message
                else:
                    feedback = "end"

                if feedback == "end":
                    break
                # [Query]:xxxx<SEP>[Answer]:xxxx<SEP>[Reference]:xxxx<SEP>
                feedback_list = feedback.split("<SEP>")
                if "Unsolved Query" not in feedback:
                    new_prompt = """Reference: {} According to this Reference, the answer for "{}" should be "{}", you can change your answer based on the Reference and continue constructing the reasoning chain to give the final answer for [Question]:{}""".format(
                        feedback_list[0], feedback_list[1], q, feedback_list[2]
                    )
                else:
                    new_prompt = """Reference: {} According to this Reference, the answer for "{}" should be "{}", you can give your answer based on the Reference and continue constructing the reasoning chain to give the final answer for [Question]：{} """.format(
                        feedback_list[0], feedback_list[1], q, feedback_list[2]
                    )
                message_keys_list.append(ChatMessage(role="user", content=new_prompt))
            result = self._extract(message_keys_list)
            print(result)

        return -1
