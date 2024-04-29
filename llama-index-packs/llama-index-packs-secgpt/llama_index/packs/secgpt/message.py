"""
In SecGPT, all messages exchanged among spokes conform to predefined formats, encapsulated within the Message class.
"""
import json


class Message:
    def function_probe_request(self, spoke_id, function):
        message = {}
        message["message_type"] = "function_probe_request"
        message["spoke_id"] = spoke_id
        message["requested_functionality"] = function  # functionality name str
        return json.dumps(message).encode("utf-8")

    def function_probe_response(self, spoke_id, function):
        message = {}
        message["message_type"] = "function_probe_response"
        message["spoke_id"] = spoke_id
        message["functionality_offered"] = function  # should be a json format
        return json.dumps(message).encode("utf-8")

    def app_request(self, spoke_id, function, functionality_request):
        message = {}
        message["message_type"] = "app_request"
        message["spoke_id"] = spoke_id
        message["functionality_request"] = function
        message["request_body"] = functionality_request  # format the request with json
        return json.dumps(message).encode("utf-8")

    def app_response(self, spoke_id, functionality_response):
        message = {}
        message["message_type"] = "app_response"
        message["spoke_id"] = spoke_id
        message["response"] = functionality_response
        return json.dumps(message).encode("utf-8")

    def final_response(self, spoke_id, final_response):
        message = {}
        message["message_type"] = "final_response"
        message["spoke_id"] = spoke_id
        message["response"] = final_response
        return json.dumps(message).encode("utf-8")

    def no_functionality_response(self, spoke_id, functionality_request):
        message = {}
        message["message_type"] = "no_functionality_response"
        message["spoke_id"] = spoke_id
        message["response"] = functionality_request + " not found"
        return json.dumps(message).encode("utf-8")

    def functionality_denial_response(self, spoke_id, functionality_request):
        message = {}
        message["message_type"] = "functionality_denial_response"
        message["spoke_id"] = spoke_id
        message["response"] = functionality_request + " refuses to respond"
        return json.dumps(message).encode("utf-8")
