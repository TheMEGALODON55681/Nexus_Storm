import os
import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Union, Optional
from pathlib import Path

try:
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        if "LITELLM_LOCAL_MODEL_COST_MAP" not in os.environ:
            os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        import litellm

        litellm.drop_params = True
        litellm.telemetry = False

    from litellm.caching.caching import Cache

    _cache_dir = os.path.join(Path.home(), ".storm_local_cache")
    litellm.cache = Cache(disk_cache_dir=_cache_dir, type="disk")

except ImportError:

    class _LitellmStub:
        def __getattr__(self, _):
            raise ImportError(
                "LiteLLM is not installed. Run `pip install litellm` to enable embedding support."
            )

    litellm = _LitellmStub()


class Encoder:
    """
    Embedding wrapper built on LiteLLM.

    Supports parallel embedding generation and transparent disk caching via
    LiteLLM's built-in cache layer. Tracks cumulative token usage for cost monitoring.

    Supported encoder types: ``'openai'``, ``'azure'``.
    For a full list of compatible models see:
    https://docs.litellm.ai/docs/embedding/supported_embedding
    """

    def __init__(
        self,
        encoder_type: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
    ):
        """
        Initialise the Encoder.

        Args:
            encoder_type: Provider type (``'openai'`` or ``'azure'``).
                Falls back to the ``ENCODER_API_TYPE`` environment variable.
            api_key: Provider API key (falls back to the relevant env var).
            api_base: Base URL for API calls (Azure only).
            api_version: API version string (Azure only).
        """
        self.embedding_model_name: Optional[str] = None
        self.kargs: dict = {}
        self.total_token_usage: int = 0

        resolved_type = encoder_type or os.getenv("ENCODER_API_TYPE")
        if not resolved_type:
            raise ValueError(
                "No encoder type provided. Set ENCODER_API_TYPE or pass encoder_type explicitly."
            )

        if resolved_type.lower() == "openai":
            self.embedding_model_name = "text-embedding-3-small"
            self.kargs = {"api_key": api_key or os.getenv("OPENAI_API_KEY")}
        elif resolved_type.lower() == "azure":
            self.embedding_model_name = "azure/text-embedding-3-small"
            self.kargs = {
                "api_key": api_key or os.getenv("AZURE_API_KEY"),
                "api_base": api_base or os.getenv("AZURE_API_BASE"),
                "api_version": api_version or os.getenv("AZURE_API_VERSION"),
            }
        else:
            raise ValueError(
                f"Unsupported encoder type '{resolved_type}'. "
                "Supported options are: 'openai', 'azure'."
            )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_total_token_usage(self, reset: bool = False) -> int:
        """
        Return the cumulative token count since initialisation (or last reset).

        Args:
            reset: If True, zero the counter after returning its value.
        """
        usage = self.total_token_usage
        if reset:
            self.total_token_usage = 0
        return usage

    def encode(self, texts: Union[str, List[str]], max_workers: int = 5) -> np.ndarray:
        """
        Generate embeddings for one or more text strings.

        Args:
            texts: A single string or a list of strings to embed.
            max_workers: Thread pool size for parallel embedding calls.

        Returns:
            A 1-D or 2-D numpy array of embedding vectors.
        """
        return self._embed_texts(texts, max_workers=max_workers)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _embed_single(self, text: str) -> Tuple[str, list, int]:
        """Call the embedding API for a single text string."""
        response = litellm.embedding(
            model=self.embedding_model_name, input=text, caching=True, **self.kargs
        )
        vector = response.data[0]["embedding"]
        tokens = response.get("usage", {}).get("total_tokens", 0)
        return text, vector, tokens

    def _embed_texts(
        self,
        texts: Union[str, List[str]],
        max_workers: int = 5,
    ) -> np.ndarray:
        """
        Embed one or more texts, using parallel calls for lists.

        Input order is preserved in the returned array.
        """
        if isinstance(texts, str):
            _, vector, tokens = self._embed_single(texts)
            self.total_token_usage += tokens
            return np.array(vector)

        batch_results = []
        total_tokens = 0

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self._embed_single, text): text for text in texts}
            for future in as_completed(futures):
                try:
                    text, vector, tokens = future.result()
                    batch_results.append((text, vector, tokens))
                    total_tokens += tokens
                except Exception as exc:
                    print(f"Embedding failed for text: {futures[future]!r}\n{exc}")

        # Restore original ordering.
        batch_results.sort(key=lambda x: texts.index(x[0]))
        self.total_token_usage += total_tokens

        return np.array([r[1] for r in batch_results])
