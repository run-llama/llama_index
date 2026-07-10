import sys
import os

sys.path.insert(0, os.path.abspath('llama-index-core'))

from llama_index.core.schema import ImageDocument
from llama_index.core.multi_modal_llms.generic_utils import image_documents_to_base64

def test_bypass():
    print("Testando leitura de /etc/passwd...")
    doc = ImageDocument(metadata={"file_path": "/etc/passwd"})
    try:
        encoded = image_documents_to_base64([doc])
        print(f"FAILED: O exploit funcionou e retornou base64: {encoded[0][:30]}...")
    except ValueError as e:
        print(f"SUCCESS: Ataque bloqueado com sucesso! Erro: {e}")
    except Exception as e:
        print(f"ERROR: Outro erro ocorreu: {e}")

if __name__ == "__main__":
    test_bypass()
