import os
import tempfile

# Must be set before the app package is imported.
_tmp = tempfile.mkdtemp(prefix="chroma_test_")
os.environ["CHROMA_DIR"] = _tmp
os.environ["KNOWLEDGE_DIR"] = os.path.join(_tmp, "empty_kb")
os.environ["ADMIN_KEY"] = "test-admin"
os.environ["GEMINI_API_KEY"] = "fake-key-for-tests"
