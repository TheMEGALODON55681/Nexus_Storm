import concurrent.futures
import copy
import logging
from concurrent.futures import as_completed
from typing import List, Union

import dspy

from .callback import BaseCallbackHandler
from .storm_dataclass import StormInformationTable, StormArticle
from ...interface import ArticleGenerationModule, Information
from ...utils import ArticleTextProcessing


# --------------------------------------------------------------------------- #
# DSPy Signature                                                               #
# --------------------------------------------------------------------------- #

class WriteSection(dspy.Signature):
    """Write a single Wikipedia article section using the provided source material.

    Formatting requirements:
    1. Begin with the section heading using '#', '##', '###', etc. for the appropriate depth.
    2. Inline-cite every factual claim using bracket notation, e.g. [1], [2], [3].
    3. Do not add a References or Sources section at the end.
    """

    info = dspy.InputField(prefix="Source material:\n", format=str)
    topic = dspy.InputField(prefix="Article topic: ", format=str)
    section = dspy.InputField(prefix="Section to write: ", format=str)
    output = dspy.OutputField(
        prefix=(
            "Write the section with inline citations. "
            "Start with the section heading (e.g. '# Section Title'). "
            "Do not write other sections or repeat the article title.\n"
        ),
        format=str,
    )


# --------------------------------------------------------------------------- #
# DSPy Module                                                                  #
# --------------------------------------------------------------------------- #

class ConvToSection(dspy.Module):
    """Converts aggregated source snippets into a polished article section."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.write_section = dspy.Predict(WriteSection)
        self.engine = engine

    def forward(
        self, topic: str, outline: str, section: str, collected_info: List[Information]
    ):
        # Combine snippets from all retrieved sources into a numbered reference block.
        source_text = ""
        for idx, source in enumerate(collected_info):
            source_text += f"[{idx + 1}]\n" + "\n".join(source.snippets)
            source_text += "\n\n"

        source_text = ArticleTextProcessing.limit_word_count_preserve_newline(
            source_text, 1500
        )

        with dspy.settings.context(lm=self.engine):
            written = ArticleTextProcessing.clean_up_section(
                self.write_section(
                    topic=topic, info=source_text, section=section
                ).output
            )

        return dspy.Prediction(section=written)


# --------------------------------------------------------------------------- #
# Article Generation Module                                                    #
# --------------------------------------------------------------------------- #

class StormArticleGenerationModule(ArticleGenerationModule):
    """
    Fills in a structured article outline with substantive, cited content.

    Each non-trivial section is written in parallel using the provided LM and
    the information gathered during the curation stage.
    """

    # Sections matching these keywords are skipped — they are handled separately
    # (introduction via polishing; conclusions are intentionally omitted).
    _SKIP_SECTION_PREFIXES = ("introduction", "conclusion", "summary")

    def __init__(
        self,
        article_gen_lm=Union[dspy.dsp.LM, dspy.dsp.HFModel],
        retrieve_top_k: int = 5,
        max_thread_num: int = 10,
    ):
        super().__init__()
        self.retrieve_top_k = retrieve_top_k
        self.article_gen_lm = article_gen_lm
        self.max_thread_num = max_thread_num
        self.section_gen = ConvToSection(engine=self.article_gen_lm)

    def _should_skip_section(self, section_title: str) -> bool:
        normalised = section_title.lower().strip()
        return any(normalised.startswith(p) for p in self._SKIP_SECTION_PREFIXES)

    def generate_section(
        self, topic, section_name, information_table, section_outline, section_query
    ):
        """Write one section and return its output dict."""
        retrieved: List[Information] = []
        if information_table is not None:
            retrieved = information_table.retrieve_information(
                queries=section_query, search_top_k=self.retrieve_top_k
            )

        output = self.section_gen(
            topic=topic,
            outline=section_outline,
            section=section_name,
            collected_info=retrieved,
        )
        return {
            "section_name": section_name,
            "section_content": output.section,
            "collected_info": retrieved,
        }

    def generate_article(
        self,
        topic: str,
        information_table: StormInformationTable,
        article_with_outline: StormArticle,
        callback_handler: BaseCallbackHandler = None,
    ) -> StormArticle:
        """
        Populate the outline with LM-generated, source-grounded content.

        Args:
            topic: The article topic.
            information_table: Curated sources from the knowledge-curation stage.
            article_with_outline: Structured outline from the outline-generation stage.
            callback_handler: Optional progress callback handler.

        Returns:
            A StormArticle with all sections filled in.
        """
        information_table.prepare_table_for_retrieval()

        if article_with_outline is None:
            article_with_outline = StormArticle(topic_name=topic)

        top_level_sections = article_with_outline.get_first_level_section_names()
        section_results = []

        if not top_level_sections:
            logging.error(
                "No outline sections found for '%s'. Falling back to topic-level generation.",
                topic,
            )
            section_results.append(
                self.generate_section(
                    topic=topic,
                    section_name=topic,
                    information_table=information_table,
                    section_outline="",
                    section_query=[topic],
                )
            )
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_thread_num
            ) as executor:
                pending = {}
                for section_title in top_level_sections:
                    if self._should_skip_section(section_title):
                        continue

                    section_query = article_with_outline.get_outline_as_list(
                        root_section_name=section_title, add_hashtags=False
                    )
                    section_outline = "\n".join(
                        article_with_outline.get_outline_as_list(
                            root_section_name=section_title, add_hashtags=True
                        )
                    )

                    future = executor.submit(
                        self.generate_section,
                        topic,
                        section_title,
                        information_table,
                        section_outline,
                        section_query,
                    )
                    pending[future] = section_title

                for future in as_completed(pending):
                    section_results.append(future.result())

        article = copy.deepcopy(article_with_outline)
        for result in section_results:
            article.update_section(
                parent_section_name=topic,
                current_section_content=result["section_content"],
                current_section_info_list=result["collected_info"],
            )
        article.post_processing()
        return article
