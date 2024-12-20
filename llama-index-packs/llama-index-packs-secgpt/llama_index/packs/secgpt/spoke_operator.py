"""
The spoke operator is a rule-based module characterized by a clearly defined execution flow that handles communication between the spoke and the hub. To implement this functionality, we have developed a SpokeOperator class.
"""

from jsonschema import validate
import ast

from .message import Message


class SpokeOperator:
    """
    A class to handle communication and execution flow between the spoke and the hub.

    Attributes:
        functionality_list (list): A list of functionalities supported by the LLM-based system.
        spoke_id (str): The identifier for the spoke.
        child_sock (Socket): The socket used for communication with the hub.
    """

    def __init__(self, functionality_list) -> None:
        """
        Initialize the SpokeOperator with a list of functionalities.

        Args:
            functionality_list (list): A list of functionalities supported by the LLM-based system.
        """
        self.functionality_list = functionality_list
        self.spoke_id = None
        self.child_sock = None

    def parse_request(self, request):
        """
        Parse a request to extract the functionality and request body.

        Args:
            request (str): The request string in JSON format.

        Returns:
            str: A formatted request string or the original request if parsing fails.
        """
        try:
            if request.startswith("{"):
                request = ast.literal_eval(request)
                functionality = request["functionality_request"]
                request_body = request["request_body"]
                request = f"{functionality}({request_body})"
            return request

        except Exception as e:
            print(e)
            return str(request)

    def probe_functionality(self, functionality: str):
        """
        Format and send a probe message to the hub to check if using a functionality is approved.

        Args:
            functionality (str): The functionality to be probed.

        Returns:
            tuple: A tuple containing the message type and the function schema if available.
        """
        # check whether the functionality is in the functionality list
        if functionality not in self.functionality_list:
            return None

        # format the functionality probe message
        probe_message = Message.function_probe_request(self.spoke_id, functionality)

        # make request to probe functionality request format
        self.child_sock.send(probe_message)
        response = self.child_sock.recv()

        if response["message_type"] == "function_probe_response":
            function_schema = response["functionality_offered"]
        else:
            function_schema = None

        return response["message_type"], function_schema

    def make_request(self, functionality: str, request: dict):
        """
        Format and send an app request message to the hub.

        Args:
            functionality (str): The functionality to be requested.
            request (dict): The request body in dictionary format.

        Returns:
            tuple: A tuple containing the message type and the response from the hub.
        """
        # format the app request message
        app_request_message = Message.app_request(self.spoke_id, functionality, request)
        self.child_sock.send(app_request_message)
        response = self.child_sock.recv()

        return response["message_type"], response["response"]

    def check_format(self, format, instance_dict):
        """
        Check if the given instance dictionary matches the specified format/schema.

        Args:
            format (dict): The JSON schema format.
            instance_dict (dict): The instance dictionary to be validated.

        Returns:
            bool: True if the instance matches the format, False otherwise.
        """
        try:
            validate(instance=instance_dict, schema=format)
            return True
        except Exception:
            return False

    def return_response(self, results):
        """
        Format and send the final response message to the hub.

        Args:
            results (dict): The results to be included in the final response.
        """
        response = Message.final_response(self.spoke_id, results)
        self.child_sock.send(response)
