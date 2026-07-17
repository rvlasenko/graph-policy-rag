import pytest

from graph_policy_rag.application.query_request import QueryRequest


def test_query_request_rejects_empty_text():
    with pytest.raises(ValueError):
        QueryRequest(
            text="",
            top_k=5,
        )
