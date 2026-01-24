import pytest


@pytest.fixture
def queries():
    return [
        {"query": "apple", "relevant_doc_ids": ["1"]},
        {"query": "orange", "relevant_doc_ids": ["2"]},
    ]


@pytest.fixture
def sample_baseline():
    return {
        "metrics": {
            "ndcg@10": 0.5,
        }
    }


@pytest.fixture
def sample_candidate():
    return {
        "metrics": {
            "ndcg@10": 0.3,
        }
    }
