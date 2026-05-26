import concurrent.futures
import dspy
import functools
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, List, Optional, Union, TYPE_CHECKING

from .utils import ArticleTextProcessing

logging.basicConfig(
    level=logging.INFO, format="%(name)s : %(levelname)-8s : %(message)s"
)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .logging_wrapper import LoggingWrapper


class InformationTable(ABC):
    """
    Base data container for information gathered during the KnowledgeCuration phase.

    Subclass this to attach additional metadata as needed. For example, the STORM
    paper (https://arxiv.org/pdf/2402.14207.pdf) stores perspective-guided dialogue history here.
    """

    def __init__(self):
        pass

    @abstractmethod
    def retrieve_information(**kwargs):
        pass


class Information:
    """Represents a discrete unit of sourced information.

    Tracks a unique URL identifier along with descriptive metadata, extracted
    snippets, and an optional citation reference number.

    Attributes:
        description (str): Short description of the source.
        snippets (list): Extracted text excerpts.
        title (str): Headline or title of the source.
        url (str): Canonical URL, used as a unique identifier.
        meta (dict): Auxiliary metadata (e.g., originating query).
        citation_uuid (int): Assigned citation index; -1 when unset.
    """

    def __init__(self, url, description, snippets, title, meta=None):
        """
        Args:
            url (str): Unique URL identifier for this information source.
            description (str): Human-readable description of the source.
            snippets (list): List of text excerpts from the source.
            title (str): Title or headline of the source.
            meta (dict, optional): Extra metadata dictionary.
        """
        self.description = description
        self.snippets = snippets
        self.title = title
        self.url = url
        self.meta = meta if meta is not None else {}
        self.citation_uuid = -1

    def __eq__(self, other):
        if not isinstance(other, Information):
            return False
        return (
            self.url == other.url
            and set(self.snippets) == set(other.snippets)
            and self._meta_str() == other._meta_str()
        )

    def __hash__(self):
        return int(
            self._md5_hash((self.url, tuple(sorted(self.snippets)), self._meta_str())),
            16,
        )

    def _meta_str(self):
        """Produce a deterministic string from the relevant meta fields."""
        return f"Question: {self.meta.get('question', '')}, Query: {self.meta.get('query', '')}"

    def _md5_hash(self, value):
        """Return an MD5 hex digest for any hashable or JSON-serialisable value."""
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, sort_keys=True)
        return hashlib.md5(str(value).encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, info_dict):
        """Construct an Information instance from a plain dictionary.

        Args:
            info_dict (dict): Must contain 'url', 'description', 'snippets', and 'title'.

        Returns:
            Information: A fully initialised instance.
        """
        instance = cls(
            url=info_dict["url"],
            description=info_dict["description"],
            snippets=info_dict["snippets"],
            title=info_dict["title"],
            meta=info_dict.get("meta", None),
        )
        instance.citation_uuid = int(info_dict.get("citation_uuid", -1))
        return instance

    def to_dict(self):
        return {
            "url": self.url,
            "description": self.description,
            "snippets": self.snippets,
            "title": self.title,
            "meta": self.meta,
            "citation_uuid": self.citation_uuid,
        }


class ArticleSectionNode:
    """
    Tree node representing a single section within an article hierarchy.

    Stores the section heading, its textual content, child sub-sections,
    and any writer preferences attached to this node.
    """

    def __init__(self, section_name: str, content=None):
        """
        Args:
            section_name (str): The heading label for this section (e.g. 'Introduction').
            content: Optional textual or structured content for this section.
        """
        self.section_name = section_name
        self.content = content
        self.children = []
        self.preference = None

    def add_child(self, new_child_node, insert_to_front=False):
        if insert_to_front:
            self.children.insert(0, new_child_node)
        else:
            self.children.append(new_child_node)

    def remove_child(self, child):
        self.children.remove(child)


class Article(ABC):
    def __init__(self, topic_name):
        self.root = ArticleSectionNode(topic_name)

    def find_section(
        self, node: ArticleSectionNode, name: str
    ) -> Optional[ArticleSectionNode]:
        """
        Depth-first search for a section node by name.

        Args:
            node: Root node to search from.
            name: Target section name.

        Returns:
            The matching ArticleSectionNode, or None if not found.
        """
        if node.section_name == name:
            return node
        for child in node.children:
            result = self.find_section(child, name)
            if result:
                return result
        return None

    @abstractmethod
    def to_string(self) -> str:
        """Serialise the article to a plain string."""

    def get_outline_tree(self):
        """
        Build a nested dictionary representing the document's section hierarchy.

        Returns:
            Dict[str, Dict]: Nested dict where keys are section names and values
                are sub-dicts of child sections (empty dict for leaf nodes).

        Example::

            {
                'Introduction': {'Background': {}, 'Objective': {}},
                'Methods': {'Data Collection': {}, 'Analysis': {}},
            }
        """

        def build_tree(node) -> Dict[str, Dict]:
            tree = {}
            for child in node.children:
                tree[child.section_name] = build_tree(child)
            return tree if tree else {}

        return build_tree(self.root)

    def get_first_level_section_names(self) -> List[str]:
        """Return the names of all top-level sections."""
        return [node.section_name for node in self.root.children]

    @classmethod
    @abstractmethod
    def from_string(cls, topic_name: str, article_text: str):
        """Create an Article instance by parsing a plain-text representation."""
        pass

    def prune_empty_nodes(self, node=None):
        if node is None:
            node = self.root

        node.children[:] = [
            child for child in node.children if self.prune_empty_nodes(child)
        ]

        if (node.content is None or node.content == "") and not node.children:
            return None
        return node


class Retriever:
    """
    Wraps a retrieval model (dspy.Retrieve) to fetch Information objects.

    Concrete retrieval/search integrations are wired up via the ``rm`` parameter.
    The attribute name for any retrieval model must end with ``_rm``.
    """

    def __init__(self, rm: dspy.Retrieve, max_thread: int = 1):
        self.max_thread = max_thread
        self.rm = rm

    def collect_and_reset_rm_usage(self):
        combined_usage = []
        if hasattr(getattr(self, "rm"), "get_usage_and_reset"):
            combined_usage.append(getattr(self, "rm").get_usage_and_reset())

        usage_by_name = {}
        for usage in combined_usage:
            for model_name, query_cnt in usage.items():
                if model_name not in usage_by_name:
                    usage_by_name[model_name] = query_cnt
                else:
                    usage_by_name[model_name] += query_cnt

        return usage_by_name

    def retrieve(
        self, query: Union[str, List[str]], exclude_urls: List[str] = []
    ) -> List[Information]:
        queries = query if isinstance(query, list) else [query]
        results_accumulator = []

        def process_query(q):
            retrieved_data_list = self.rm(
                query_or_queries=[q], exclude_urls=exclude_urls
            )
            local_results = []
            for data in retrieved_data_list:
                for i in range(len(data["snippets"])):
                    # Strip nested citations to avoid reference confusion in generated articles.
                    data["snippets"][i] = ArticleTextProcessing.remove_citations(
                        data["snippets"][i]
                    )
                info_item = Information.from_dict(data)
                info_item.meta["query"] = q
                local_results.append(info_item)
            return local_results

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_thread
        ) as executor:
            all_results = list(executor.map(process_query, queries))

        for batch in all_results:
            results_accumulator.extend(batch)

        return results_accumulator


class KnowledgeCurationModule(ABC):
    """
    Abstract interface for the knowledge curation stage.

    Implementations receive a topic string and return a populated InformationTable.
    """

    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    @abstractmethod
    def research(self, topic) -> InformationTable:
        """
        Gather and organise information about the given topic.

        Args:
            topic (str): Topic of interest expressed in natural language.

        Returns:
            InformationTable: Curated information ready for downstream stages.
        """
        pass


class OutlineGenerationModule(ABC):
    """
    Abstract interface for the outline generation stage.

    Takes the topic and curated information and produces a structured article outline.
    """

    @abstractmethod
    def generate_outline(
        self, topic: str, information_table: InformationTable, **kwargs
    ) -> Article:
        """
        Produce a structured outline for the article.

        Args:
            topic (str): The article topic.
            information_table (InformationTable): Data from the curation stage.

        Returns:
            Article: An article skeleton populated with section headings.
        """
        pass


class ArticleGenerationModule(ABC):
    """
    Abstract interface for the article generation stage.

    Fills in a structured outline with content drawn from the information table.
    """

    @abstractmethod
    def generate_article(
        self,
        topic: str,
        information_table: InformationTable,
        article_with_outline: Article,
        **kwargs,
    ) -> Article:
        """
        Populate the article outline with substantive content.

        Args:
            topic (str): The article topic.
            information_table (InformationTable): Curated source information.
            article_with_outline (Article): The outline produced by OutlineGenerationModule.

        Returns:
            Article: A fully drafted article.
        """
        pass


class ArticlePolishingModule(ABC):
    """
    Abstract interface for the article polishing stage.

    Refines a draft article by adding a lead section and optionally removing duplicates.
    """

    @abstractmethod
    def polish_article(self, topic: str, draft_article: Article, **kwargs) -> Article:
        """
        Refine and enhance a draft article.

        Args:
            topic (str): The article topic.
            draft_article (Article): The draft from ArticleGenerationModule.

        Returns:
            Article: The polished, publication-ready article.
        """
        pass


def log_execution_time(func):
    """Decorator that logs wall-clock execution time for a method."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        t_start = time.time()
        result = func(self, *args, **kwargs)
        elapsed = time.time() - t_start
        logger.info(f"{func.__name__} completed in {elapsed:.4f}s")
        self.time[func.__name__] = elapsed
        return result

    return wrapper


class LMConfigs(ABC):
    """Abstract base class for pipeline language-model configuration bundles.

    Any attribute whose name ends in ``_lm`` is treated as a language model slot
    and is included in usage collection and validation checks.
    """

    def __init__(self):
        pass

    def init_check(self):
        for attr_name in self.__dict__:
            if "_lm" in attr_name and getattr(self, attr_name) is None:
                logging.warning(
                    f"Language model for {attr_name} is not initialised. "
                    f"Call set_{attr_name}() before running the pipeline."
                )

    def collect_and_reset_lm_history(self):
        history = []
        for attr_name in self.__dict__:
            if "_lm" in attr_name and hasattr(getattr(self, attr_name), "history"):
                history.extend(getattr(self, attr_name).history)
                getattr(self, attr_name).history = []
        return history

    def collect_and_reset_lm_usage(self):
        combined_usage = []
        for attr_name in self.__dict__:
            if "_lm" in attr_name and hasattr(
                getattr(self, attr_name), "get_usage_and_reset"
            ):
                combined_usage.append(getattr(self, attr_name).get_usage_and_reset())

        usage_by_model = {}
        for usage in combined_usage:
            for model_name, token_counts in usage.items():
                if model_name not in usage_by_model:
                    usage_by_model[model_name] = token_counts
                else:
                    usage_by_model[model_name]["prompt_tokens"] += token_counts[
                        "prompt_tokens"
                    ]
                    usage_by_model[model_name]["completion_tokens"] += token_counts[
                        "completion_tokens"
                    ]

        return usage_by_model

    def log(self):
        return OrderedDict(
            {
                attr_name: getattr(self, attr_name).kwargs
                for attr_name in self.__dict__
                if "_lm" in attr_name and hasattr(getattr(self, attr_name), "kwargs")
            }
        )


class Engine(ABC):
    def __init__(self, lm_configs: LMConfigs):
        self.lm_configs = lm_configs
        self.time = {}
        self.lm_cost = {}  # Token-based cost per pipeline stage.
        self.rm_cost = {}  # Query-count-based cost per pipeline stage.

    def log_execution_time_and_lm_rm_usage(self, func):
        """Decorator: records wall time, LM usage, and RM usage for pipeline methods."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t_start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - t_start
            self.time[func.__name__] = elapsed
            logger.info(f"{func.__name__} completed in {elapsed:.4f}s")
            self.lm_cost[func.__name__] = self.lm_configs.collect_and_reset_lm_usage()
            if hasattr(self, "retriever"):
                self.rm_cost[func.__name__] = (
                    self.retriever.collect_and_reset_rm_usage()
                )
            return result

        return wrapper

    def apply_decorators(self):
        """Attach the logging/usage decorator to every method prefixed with ``run_``."""
        run_methods = [
            name
            for name in dir(self)
            if callable(getattr(self, name)) and name.startswith("run_")
        ]
        for method_name in run_methods:
            original = getattr(self, method_name)
            decorated = self.log_execution_time_and_lm_rm_usage(original)
            setattr(self, method_name, decorated)

    @abstractmethod
    def run_knowledge_curation_module(self, **kwargs) -> Optional[InformationTable]:
        pass

    @abstractmethod
    def run_outline_generation_module(self, **kwarg) -> Article:
        pass

    @abstractmethod
    def run_article_generation_module(self, **kwarg) -> Article:
        pass

    @abstractmethod
    def run_article_polishing_module(self, **kwarg) -> Article:
        pass

    @abstractmethod
    def run(self, **kwargs):
        pass

    def summary(self):
        print("***** Execution Time *****")
        for stage, duration in self.time.items():
            print(f"  {stage}: {duration:.4f}s")

        print("***** Language Model Token Usage *****")
        for stage, usage in self.lm_cost.items():
            print(f"  {stage}:")
            for model_name, tokens in usage.items():
                print(f"    {model_name}: {tokens}")

        print("***** Retrieval Model Query Counts *****")
        for stage, counts in self.rm_cost.items():
            print(f"  {stage}: {counts}")

    def reset(self):
        self.time = {}
        self.lm_cost = {}
        self.rm_cost = {}


class Agent(ABC):
    """
    Base interface for STORM and Co-STORM conversational agents.

    Each agent is characterised by a topic, a role name, and a role description.
    Subclasses implement :meth:`generate_utterance` to define how the agent
    contributes to the ongoing discourse.

    Args:
        topic (str): The subject of the current research session.
        role_name (str): Short label identifying the agent's role.
        role_description (str): Extended description of the agent's focus or persona.
    """

    from .dataclass import KnowledgeBase, ConversationTurn

    def __init__(self, topic: str, role_name: str, role_description: str):
        self.topic = topic
        self.role_name = role_name
        self.role_description = role_description

    def get_role_description(self):
        if self.role_description:
            return f"{self.role_name}: {self.role_description}"
        return self.role_name

    @abstractmethod
    def generate_utterance(
        self,
        knowledge_base: "KnowledgeBase",
        conversation_history: List["ConversationTurn"],
        logging_wrapper: "LoggingWrapper",
        **kwargs,
    ):
        pass
