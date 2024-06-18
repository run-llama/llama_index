"""
In SecGPT, all messages exchanged among spokes conform to predefined formats, encapsulated within the Message class.
"""
import json


class Message:
    @staticmethod
    def function_probe_request(spoke_id, function):
        """
        Create a function probe request message.

        Args:
            spoke_id (str): The ID of the spoke sending the request.
            function (str): The functionality being requested.

        Returns:
            bytes: The JSON-encoded function probe request message.
        """
        message = {}
        message["message_type"] = "function_probe_request"
        message["spoke_id"] = spoke_id
        message["requested_functionality"] = function  # functionality name str
        return json.dumps(message).encode("utf-8")

    @staticmethod
    def function_probe_response(spoke_id, function):
        """
        Create a function probe response message.

        Args:
            spoke_id (str): The ID of the spoke sending the response.
            function (str): The functionality being offered (in JSON format).

        Returns:
            bytes: The JSON-encoded function probe response message.
        """
        message = {}
        message["message_type"] = "function_probe_response"
        message["spoke_id"] = spoke_id
        message["functionality_offered"] = function  # should be a json format
        return json.dumps(message).encode("utf-8")

    @staticmethod
    def app_request(spoke_id, function, functionality_request):
        """
        Create an application request message.

        Args:
            spoke_id (str): The ID of the spoke sending the request.
            function (str): The functionality being requested.
            functionality_request (str): The request body formatted in JSON.

        Returns:
            bytes: The JSON-encoded application request message.
        """
        message = {}
        message["message_type"] = "app_request"
        message["spoke_id"] = spoke_id
        message["functionality_request"] = function
        message["request_body"] = functionality_request  # format the request with json
        return json.dumps(message).encode("utf-8")

    @staticmethod
    def app_response(spoke_id, functionality_response):
        """
        Create an application response message.

        Args:
            spoke_id (str): The ID of the spoke sending the response.
            functionality_response (str): The response body.

        Returns:
            bytes: The JSON-encoded application response message.
        """
        message = {}
        message["message_type"] = "app_response"
        message["spoke_id"] = spoke_id
        message["response"] = functionality_response
        return json.dumps(message).encode("utf-8")

    @staticmethod
    def final_response(spoke_id, final_response):
        """
        Create a final response message.

        Args:
            spoke_id (str): The ID of the spoke sending the final response.
            final_response (str): The final response body.

        Returns:
            bytes: The JSON-encoded final response message.
        """
        message = {}
        message["message_type"] = "final_response"
        message["spoke_id"] = spoke_id
        message["response"] = final_response
        return json.dumps(message).encode("utf-8")

    @staticmethod
    def no_functionality_response(spoke_id, functionality_request):
        """
        Create a no functionality response message indicating the requested functionality was not found.

        Args:
            spoke_id (str): The ID of the spoke sending the response.
            functionality_request (str): The functionality request that was not found.

        Returns:
            bytes: The JSON-encoded no functionality response message.
        """
        message = {}
        message["message_type"] = "no_functionality_response"
        message["spoke_id"] = spoke_id
        message["response"] = functionality_request + " not found"
        return json.dumps(message).encode("utf-8")

    @staticmethod
    def functionality_denial_response(spoke_id, functionality_request):
        """
        Create a functionality denial response message indicating the requested functionality refuses to respond.

        Args:
            spoke_id (str): The ID of the spoke sending the response.
            functionality_request (str): The functionality request that is being denied.

        Returns:
            bytes: The JSON-encoded functionality denial response message.
        """
        message = {}
        message["message_type"] = "functionality_denial_response"
        message["spoke_id"] = spoke_id
        message["response"] = functionality_request + " refuses to respond"
        return json.dumps(message).encode("utf-8")
