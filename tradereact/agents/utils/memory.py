import os
from openai import OpenAI
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery

from tradereact.default_config import DEFAULT_CONFIG


class FinancialSituationMemory:
    def __init__(self, name, config):
        # if config["backend_url"] == "http://localhost:11434/v1":
        #     self.embedding = "nomic-embed-text"
        # else:
        #     self.embedding = "text-embedding-3-small"
        self.embedding = "text-embedding-3-small"

        # ========== OpenAI Client Configuration (for embeddings) ==========
        openai_base_url = (
            config.get("backend_url") or os.getenv("OPENAI_BASE_URL") or ""
        ).strip().strip("`").strip()
        openai_api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        openai_client_kwargs = {}
        if openai_base_url:
            openai_client_kwargs["base_url"] = openai_base_url
        if openai_api_key:
            openai_client_kwargs["api_key"] = openai_api_key
        self.client = OpenAI(**openai_client_kwargs)

        # ========== Weaviate Client Configuration ==========
        # Get Weaviate connection parameters from environment variables
        weaviate_url = os.getenv("WEAVIATE_URL", "").strip()
        weaviate_api_key = os.getenv("WEAVIATE_API_KEY", "").strip()

        # Choose connection mode based on configuration
        if weaviate_url:
            # Remote Weaviate instance (cloud or self-hosted)
            if weaviate_api_key:
                # With API key authentication
                self.weaviate_client = weaviate.connect_to_wcs(
                    cluster_url=weaviate_url,
                    auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key),
                )
            else:
                # Without authentication (for local dev servers)
                self.weaviate_client = weaviate.connect_to_custom(
                    http_host=weaviate_url.replace("http://", "").replace("https://", ""),
                    http_port=80,
                    http_secure=weaviate_url.startswith("https"),
                    grpc_host=weaviate_url.replace("http://", "").replace("https://", ""),
                    grpc_port=50051,
                    grpc_secure=weaviate_url.startswith("https"),
                )
        else:
            # Embedded mode (local development - no server needed)
            self.weaviate_client = weaviate.connect_to_embedded()

        # Normalize collection name (Weaviate requires PascalCase class names)
        self.collection_name = "".join(word.capitalize() for word in name.split("_"))

        # Create collection schema if it doesn't exist
        if not self.weaviate_client.collections.exists(self.collection_name):
            self.weaviate_client.collections.create(
                name=self.collection_name,
                properties=[
                    Property(name="situation", data_type=DataType.TEXT),
                    Property(name="recommendation", data_type=DataType.TEXT),
                ],
                vectorizer_config=Configure.Vectorizer.none(),  # We provide our own vectors
            )

        self.collection = self.weaviate_client.collections.get(self.collection_name)

    def get_embedding(self, text):
        """Get OpenAI embedding for a text"""
        
        response = self.client.embeddings.create(
            model=self.embedding, input=text
        )
        return response.data[0].embedding

    def add_situations(self, situations_and_advice):
        """Add financial situations and their corresponding advice. Parameter is a list of tuples (situation, rec)"""

        # Prepare data objects for batch insertion
        data_objects = []

        for situation, recommendation in situations_and_advice:
            # Get embedding for the situation
            vector = self.get_embedding(situation)

            # Create data object with properties and vector
            data_objects.append({
                "properties": {
                    "situation": situation,
                    "recommendation": recommendation,
                },
                "vector": vector,
            })

        # Batch insert into Weaviate
        with self.collection.batch.dynamic() as batch:
            for obj in data_objects:
                batch.add_object(
                    properties=obj["properties"],
                    vector=obj["vector"],
                )

    def get_memories(self, current_situation, n_matches=1, alpha=0.5):
        """
        Find matching recommendations using hybrid search (BM25 + vector similarity)

        Args:
            current_situation: Query text describing the current financial situation
            n_matches: Number of top matches to return
            alpha: Hybrid search balance (0.0 = pure BM25, 1.0 = pure vector, 0.5 = balanced)

        Returns:
            List of matched results with situation, recommendation, and similarity score
        """
        query_embedding = self.get_embedding(current_situation)

        # Perform hybrid search combining BM25 (keyword) and vector similarity
        response = self.collection.query.hybrid(
            query=current_situation,
            vector=query_embedding,
            alpha=alpha,  # Balance between BM25 and vector search
            limit=n_matches,
            return_metadata=MetadataQuery(score=True, distance=True),
        )

        # Format results to match original interface
        matched_results = []
        for obj in response.objects:
            matched_results.append(
                {
                    "matched_situation": obj.properties["situation"],
                    "recommendation": obj.properties["recommendation"],
                    "similarity_score": obj.metadata.score if obj.metadata.score is not None else 0.0,
                }
            )

        return matched_results


if __name__ == "__main__":

    # Example usage
    matcher = FinancialSituationMemory(name="bull_memory", config=DEFAULT_CONFIG)

    # Example data
    example_data = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows.",
        ),
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. Consider increasing exposure to sectors benefiting from higher rates.",
        ),
    ]

    # Add the example situations and recommendations
    matcher.add_situations(example_data)

    # Example query
    current_situation = """
    Market showing increased volatility in tech sector, with institutional investors 
    reducing positions and rising interest rates affecting growth stock valuations
    """

    try:
        recommendations = matcher.get_memories(current_situation, n_matches=2)
        print(recommendations)

        for i, rec in enumerate(recommendations, 1):
            print(f"\nMatch {i}:")
            print(f"Similarity Score: {rec['similarity_score']:.2f}")
            print(f"Matched Situation: {rec['matched_situation']}")
            print(f"Recommendation: {rec['recommendation']}")

    except Exception as e:
        print(f"Error during recommendation: {str(e)}")
