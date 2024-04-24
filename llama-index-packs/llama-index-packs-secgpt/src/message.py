import json

class Message:
    def function_probe_request(self, spoke_id, function):
        message = dict()
        message['message_type'] = 'function_probe_request' 
        message['spoke_id'] = spoke_id
        message['requested_functionality'] = function # functionality name str
        serialized_msg = json.dumps(message).encode('utf-8')
        return serialized_msg
    
    def function_probe_response(self, spoke_id, function):
        message = dict()
        message['message_type'] = 'function_probe_response'
        message['spoke_id'] = spoke_id
        message['functionality_offered'] = function # should be a json format
        serialized_msg = json.dumps(message).encode('utf-8')
        return serialized_msg
    
    def app_request(self, spoke_id, function, functionality_request):
        message = dict()
        message['message_type'] = 'app_request' 
        message['spoke_id'] = spoke_id
        message['functionality_request'] = function
        message['request_body'] = functionality_request # format the request with json
        serialized_msg = json.dumps(message).encode('utf-8')
        return serialized_msg
    
    def app_response(self, spoke_id, functionality_response):
        message = dict()
        message['message_type'] = 'app_response'
        message['spoke_id'] = spoke_id
        message['response'] = functionality_response
        serialized_msg = json.dumps(message).encode('utf-8')
        return serialized_msg
    
    def final_response(self, spoke_id, final_response):
        message = dict()
        message['message_type'] = 'final_response'
        message['spoke_id'] = spoke_id
        message['response'] = final_response
        serialized_msg = json.dumps(message).encode('utf-8')
        return serialized_msg
    
    def no_functionality_response(self, spoke_id, functionality_request):
        message = dict()
        message['message_type'] = 'no_functionality_response'
        message['spoke_id'] = spoke_id
        message['response'] = functionality_request+" not found"
        serialized_msg = json.dumps(message).encode('utf-8')
        return serialized_msg

    def functionality_denial_response(self, spoke_id, functionality_request):
        message = dict()
        message['message_type'] = 'functionality_denial_response'
        message['spoke_id'] = spoke_id
        message['response'] = functionality_request+" refuses to respond"
        serialized_msg = json.dumps(message).encode('utf-8')
        return serialized_msg
    