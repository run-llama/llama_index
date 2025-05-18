from unittest.mock import MagicMock, patch

from llama_index.agent.azure_foundry_agent.base import AzureFoundryAgent
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import Agent as AzureAgent, AgentThread, ThreadRun  

def test_azure_foundry_agent_constructor():
    """Test the constructor of AzureFoundryAgent."""
    endpoint = "https://test-endpoint.com"
    model = "gpt-4o"
    name = "test-azure-agent"
    instructions = "You are a test agent."
    thread_id = "test-thread-123"
    verbose = True
    run_retrieve_sleep_time = 0.5
    
    mock_project_client_instance = MagicMock(spec=AIProjectClient)
    mock_azure_agent_instance = MagicMock(spec=AzureAgent)
    mock_azure_agent_instance.id = "mock_agent_id_123"
    mock_thread_instance = MagicMock(spec=AgentThread) # Changed Thread to AgentThread
    mock_thread_instance.id = thread_id # Ensure the mock thread has the provided ID

    # Mock DefaultAzureCredential to avoid actual credential loading
    with patch('llama_index.agent.azure_foundry_agent.base.DefaultAzureCredential', MagicMock()) as mock_default_credential:
        # Mock AIProjectClient constructor to return our mock instance
        with patch('llama_index.agent.azure_foundry_agent.base.AIProjectClient', return_value=mock_project_client_instance) as mock_ai_project_client_constructor:
            # Mock the create_agent call
            mock_project_client_instance.agents.create_agent.return_value = mock_azure_agent_instance
            # Mock the threads.create call for when thread_id is None
            mock_project_client_instance.agents.threads.create.return_value = mock_thread_instance

            # Test case 1: Initialize with a specific thread_id
            agent_with_thread = AzureFoundryAgent(
                endpoint=endpoint,
                model=model,
                name=name,
                instructions=instructions,
                thread_id=thread_id,
                verbose=verbose,
                run_retrieve_sleep_time=run_retrieve_sleep_time,
                toolset=None, # Explicitly pass None for toolset if not testing tools here
            )

            mock_ai_project_client_constructor.assert_called_once_with(
                endpoint=endpoint, credential=mock_default_credential.return_value
            )
            # Ensure threads.create was NOT called because thread_id was provided
            mock_project_client_instance.agents.threads.create.assert_not_called()
            assert isinstance(agent_with_thread, AzureFoundryAgent)
            assert agent_with_thread._endpoint == endpoint
            assert agent_with_thread._model == model
            assert agent_with_thread._name == name
            assert agent_with_thread._instructions == instructions
            assert agent_with_thread._thread_id == thread_id
            assert agent_with_thread._verbose == verbose
            assert agent_with_thread._run_retrieve_sleep_time == run_retrieve_sleep_time
            assert agent_with_thread._project_client == mock_project_client_instance
            
            # Reset mocks for the next instantiation test
            mock_ai_project_client_constructor.reset_mock()
            mock_project_client_instance.reset_mock()
            mock_default_credential.reset_mock()
            
            # Mock the threads.create call for when thread_id is None
            # Re-assign thread_id for the new mock thread instance if it's different
            new_mock_thread_id = "new-mock-thread-456"
            mock_thread_instance_new = MagicMock(spec=AgentThread) # Changed Thread to AgentThread
            mock_thread_instance_new.id = new_mock_thread_id
            mock_project_client_instance.agents.threads.create.return_value = mock_thread_instance_new

            # Test case 2: Initialize without a specific thread_id (should create one)
            agent_new_thread = AzureFoundryAgent(
                endpoint=endpoint,
                model=model,
                name=name,
                instructions=instructions,
                thread_id=None, # Test thread creation
                verbose=verbose,
                run_retrieve_sleep_time=run_retrieve_sleep_time,
                toolset=None,
            )
            
            mock_ai_project_client_constructor.assert_called_once_with(
                endpoint=endpoint, credential=mock_default_credential.return_value
            )
            # Ensure threads.create WAS called because thread_id was None
            mock_project_client_instance.agents.threads.create.assert_called_once()
            assert agent_new_thread._thread_id == new_mock_thread_id

 