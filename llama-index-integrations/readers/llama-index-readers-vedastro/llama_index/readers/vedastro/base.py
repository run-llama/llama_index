"""Simple Web scraper."""
from typing import List, Optional, Dict, Callable

import requests
import json
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document
from vedastro import *


class SimpleBirthTimeReader(BasePydanticReader):
	"""Simple birth time prediction reader.

    Reads horoscope predictions from vedastro.org
    `pip install vedastro` needed
    Args:
        metadata_fn (Optional[Callable[[str], Dict]]): A function that takes in
            a birth time and returns a dictionary of prediction metadata.
            Default is None.
    """

	is_remote: bool = True

	_metadata_fn: Optional[Callable[[str], Dict]] = PrivateAttr()

	def __init__(
	    self,
	    metadata_fn: Optional[Callable[[str], Dict]] = None,
	) -> None:
		"""Initialize with parameters."""

		self._metadata_fn = metadata_fn
		super().__init__()

	@classmethod
	def class_name(cls) -> str:
		return "SimpleBirthTimeReader"

	def load_data(self, birth_time: str) -> List[Document]:
		"""Load data from the given birth time.

        Args:
            birth_time (str): birth time in this format : Location/Delhi,India/Time/01:30/14/02/2024/+05:30

        Returns:
            List[Document]: List of documents.

        """

		documents = SimpleBirthTimeReader.birth_time_to_llama_index_nodes(birth_time)

		return documents

	@staticmethod
	# converts vedastro horoscope predictions (JSON) to_llama-index's NodeWithScore
	# so that llama index can understand vedastro predictions
	def vedastro_predictions_to_llama_index_weight_nodes(birth_time, predictions_list_json):
		from llama_index.core.schema import NodeWithScore
		from llama_index.core.schema import TextNode

		# Initialize an empty list
		prediction_nodes = []
		for prediction in predictions_list_json:

			related_bod_json = prediction['RelatedBody']

			# shadbala_score = Calculate.PlanetCombinedShadbala()
			rel_planets = related_bod_json["Planets"]
			parsed_list = []
			for planet in rel_planets:
				parsed_list.append(PlanetName.Parse(planet))

			# TODO temp use 1st planet, house, zodiac
			planet_tags = []
			shadbala_score = 0
			if parsed_list:  # This checks if the list is not empty
				for planet in parsed_list:
					shadbala_score += Calculate.PlanetShadbalaPinda(planet, birth_time).ToDouble()
					# planet_tags = Calculate.GetPlanetTags(parsed_list[0])

			predict_node = TextNode(
			    text=prediction["Description"],
			    metadata={"name": ChatTools.split_camel_case(prediction['Name'])
			              # "related_body": prediction['RelatedBody'],
			              # "planet_tags": planet_tags,
			             },
			    metadata_seperator="::",
			    metadata_template="{key}=>{value}",
			    text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
			)

			# add in shadbala to give each prediction weights
			parsed_node = NodeWithScore(node=predict_node, score=shadbala_score)  # add in shabala score
			prediction_nodes.append(parsed_node)  # add to main list

		return prediction_nodes

	@staticmethod
	def birth_time_to_llama_index_nodes(birth_time_text):

		# 1 : convert raw time text into parsed time (aka time url)
		parsed_birth_time = Time.FromUrl(birth_time_text).GetAwaiter().GetResult()

		# 2 : do +300 horoscope prediction calculations to find correct predictions for person
		all_predictions_raw = Calculate.HoroscopePredictions(parsed_birth_time)

		# show the number of horo records found
		print(f"Predictions Found : {len(all_predictions_raw)}")

		# format list nicely so LLM can swallow (llama_index nodes)
		# so that llama index can understand vedastro predictions
		all_predictions_json = json.loads(HoroscopePrediction.ToJsonList(all_predictions_raw).ToString())

		# do final packing into llama-index formats
		prediction_nodes = SimpleBirthTimeReader.vedastro_predictions_to_llama_index_documents(all_predictions_json)

		return prediction_nodes

	@staticmethod
	def vedastro_predictions_to_llama_index_nodes(birth_time, predictions_list_json):
		from llama_index.core.schema import NodeWithScore
		from llama_index.core.schema import TextNode

		# Initialize an empty list
		prediction_nodes = []
		for prediction in predictions_list_json:

			related_bod_json = prediction['RelatedBody']

			# shadbala_score = Calculate.PlanetCombinedShadbala()
			rel_planets = related_bod_json["Planets"]
			parsed_list = []
			for planet in rel_planets:
				parsed_list.append(PlanetName.Parse(planet))

			# TODO temp use 1st planet, house, zodiac
			planet_tags = []
			shadbala_score = 0
			if parsed_list:  # This checks if the list is not empty
				shadbala_score = Calculate.PlanetShadbalaPinda(parsed_list[0], birth_time).ToDouble()
				planet_tags = Calculate.GetPlanetTags(parsed_list[0])

			predict_node = TextNode(
			    text=prediction["Description"],
			    metadata={
			        "name": ChatTools.split_camel_case(prediction['Name']),
			        "related_body": prediction['RelatedBody'],
			        "planet_tags": planet_tags,
			    },
			    metadata_seperator="::",
			    metadata_template="{key}=>{value}",
			    text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
			)

			# add in shadbala to give each prediction weights
			prediction_nodes.append(predict_node)  # add to main list

		return prediction_nodes

	@staticmethod
	# given list vedastro lib horoscope predictions will convert to documents
	def vedastro_predictions_to_llama_index_documents(predictions_list_json):
		from llama_index.core import Document
		from llama_index.core.schema import MetadataMode
		import copy

		# Initialize an empty list
		prediction_nodes = []
		for prediction in predictions_list_json:

			# take out description (long text) from metadata, becasue already in as "content"
			predict_meta = copy.deepcopy(prediction)
			del predict_meta["Description"]

			predict_node = Document(
			    text=prediction["Description"],
			    metadata=predict_meta,
			    metadata_seperator="::",
			    metadata_template="{key}=>{value}",
			    text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
			)

			# # this is shows difference for understanding output of Documents
			# print("#######################################################")
			# print(
			#     "The LLM sees this: \n",
			#     predict_node.get_content(metadata_mode=MetadataMode.LLM),
			# )
			# print(
			#     "The Embedding model sees this: \n",
			#     predict_node.get_content(metadata_mode=MetadataMode.EMBED),
			# )
			# print("#######################################################")

			# add in shadbala to give each prediction weights
			prediction_nodes.append(predict_node)  # add to main list

		return prediction_nodes
