from graph_policy_rag.domain.document import Document


def build_demo_document() -> Document:
    return Document(
        id="returns-policy",
        title="Returns Policy",
        content=(
            "Customers may return unused products within 30 days of delivery. "
            "The product must remain in its original packaging. "
            "Refunds are issued to the original payment method."
        ),
        source_uri="demo://returns-policy",
        metadata={},
        version="version-1",
    )
