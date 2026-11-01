import pytest
from llama_index.core.tools.circuit_breaker import ASTCircuitBreaker

def test_circuit_breaker_safe_code():
    safe_code = """
def add(a, b):
    return a + b
    
result = add(2, 3)
print(result)
"""
    # Should not raise any exception
    ASTCircuitBreaker.validate_code(safe_code)

def test_circuit_breaker_forbidden_import():
    unsafe_code = """
import os
os.system('echo "Hacked"')
"""
    with pytest.raises(ValueError, match="AST Circuit Breaker Tripped"):
        ASTCircuitBreaker.validate_code(unsafe_code)

def test_circuit_breaker_forbidden_call():
    unsafe_code = """
eval("print('malicious')")
"""
    with pytest.raises(ValueError, match="AST Circuit Breaker Tripped"):
        ASTCircuitBreaker.validate_code(unsafe_code)