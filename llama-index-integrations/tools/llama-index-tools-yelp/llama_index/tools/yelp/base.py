"""
Copyright (c) 2013, Triad National Security, LLC
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and the following
  disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
  following disclaimer in the documentation and/or other materials provided with the distribution.
* Neither the name of Triad National Security, LLC nor the names of its contributors may be used to endorse or
  promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from typing import Optional

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class YelpToolSpec(BaseToolSpec):
    """Yelp tool spec."""

    # TODO add disclaimer
    spec_functions = ["business_search", "business_reviews"]

    def __init__(self, api_key: str, client_id: str) -> Document:
        """Initialize with parameters."""
        from yelpapi import YelpAPI

        self.client = YelpAPI(api_key)

    def business_search(self, location: str, term: str, radius: Optional[int] = None):
        """
        Make a query to Yelp to find businesses given a location to search.

        Args:
            Businesses returned in the response may not be strictly within the specified location.
            term (str): Search term, e.g. "food" or "restaurants", The term may also be the business's name, such as "Starbucks"
            radius (int): A suggested search radius in meters. This field is used as a suggestion to the search. The actual search radius may be lower than the suggested radius in dense urban areas, and higher in regions of less business density.


        """
        response = self.client.search_query(location=location, term=term)
        return [Document(text=str(response))]

    def business_reviews(self, id: str):
        """
        Make a query to Yelp to find a business using an id from business_search.

        Args:
            # The id
        """
        response = self.client.reviews_query(id=id)
        return [Document(text=str(response))]
