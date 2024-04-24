from jsonschema import validate 
import ast

from .message import Message


class SpokeOperator:
    def __init__(self, functionality_list):
        self.functionality_list = functionality_list
        self.spoke_id = None
        self.child_sock = None

    def parse_request(self, request):
        try:
            if request.startswith('{'):
                request = ast.literal_eval(request)
                functionality = request['functionality_request']
                request_body = request['request_body']
                request = f"{functionality}({request_body})"
            return request
            
        except Exception as e:
            print(e)
            return str(request)
    
    # Format and send the probe message to the hub
    def probe_functionality(self, functionality:str):
        # check whether the functionality is in the functionality list
        if functionality not in self.functionality_list:
            return
        
        # format the functionality probe message
        probe_message = Message().function_probe_request(self.spoke_id, functionality)

        # make request to probe functionality request format
        self.child_sock.send(probe_message)
        response = self.child_sock.recv()

        if response['message_type'] == 'function_probe_response':
            function_schema = response['functionality_offered']
        else:
            function_schema = None

        return response['message_type'], function_schema

    # Format and send the app request message to the hub
    def make_request(self, functionality: str, request: dict):
        # format the app request message
        app_request_message = Message().app_request(self.spoke_id, functionality, request)
        self.child_sock.send(app_request_message)
        response = self.child_sock.recv()
        
        return response['message_type'], response['response']

    def check_format(self, format, instance_dict):
        try:
            validate(instance=instance_dict, schema=format)
            return True
        except:
            return False

    def return_response(self,  results):
        response = Message().final_response(self.spoke_id, results)
        self.child_sock.send(response)
            