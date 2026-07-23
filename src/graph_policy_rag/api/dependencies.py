from functools import lru_cache

from graph_policy_rag.application.query_pipeline import QueryPipeline
from graph_policy_rag.bootstrap import (
    ApplicationContainer,
    build_application_container,
)


@lru_cache
def get_application_container() -> ApplicationContainer:
    return build_application_container()


def get_query_pipeline() -> QueryPipeline:
    return get_application_container().query_pipeline
