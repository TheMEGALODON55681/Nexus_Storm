import copy
from typing import Union

import dspy

from .storm_dataclass import StormArticle
from ...interface import ArticlePolishingModule
from ...utils import ArticleTextProcessing


# --------------------------------------------------------------------------- #
# DSPy Signatures                                                              #
# --------------------------------------------------------------------------- #

class WriteLeadSection(dspy.Signature):
    """Compose a Wikipedia-style lead section for the article.

    The lead must:
    1. Stand alone as a concise, self-contained overview of the topic.
    2. Establish context, explain the topic's significance, and surface major controversies.
    3. Consist of no more than four well-crafted paragraphs.
    4. Include inline citations where appropriate (e.g. \"The capital is Washington, D.C.[1][3]\").
    """

    topic = dspy.InputField(prefix="Article topic: ", format=str)
    draft_page = dspy.InputField(prefix="Draft article:\n", format=str)
    lead_section = dspy.OutputField(prefix="Write the lead section:\n", format=str)


class PolishPage(dspy.Signature):
    """Remove duplicate content from the article without deleting any unique information.

    Preserve all inline citations and the hierarchical section structure indicated by
    '#', '##', '###', etc. Your only task is deduplication.
    """

    draft_page = dspy.InputField(prefix="Draft article:\n", format=str)
    page = dspy.OutputField(prefix="Revised article:\n", format=str)


# --------------------------------------------------------------------------- #
# DSPy Module                                                                  #
# --------------------------------------------------------------------------- #

class PolishPageModule(dspy.Module):
    """Adds a lead section and optionally deduplicates the full article."""

    def __init__(
        self,
        write_lead_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        polish_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
    ):
        super().__init__()
        self.write_lead_engine = write_lead_engine
        self.polish_engine = polish_engine
        self.write_lead = dspy.Predict(WriteLeadSection)
        self.polish_page = dspy.Predict(PolishPage)

    def forward(self, topic: str, draft_page: str, polish_whole_page: bool = True):
        with dspy.settings.context(lm=self.write_lead_engine, show_guidelines=False):
            lead_section = self.write_lead(topic=topic, draft_page=draft_page).lead_section
            # Strip any echoed prefix that the model may have included.
            if "The lead section:" in lead_section:
                lead_section = lead_section.split("The lead section:")[1].strip()

        if polish_whole_page:
            with dspy.settings.context(lm=self.polish_engine, show_guidelines=False):
                page = self.polish_page(draft_page=draft_page).page
        else:
            page = draft_page

        return dspy.Prediction(lead_section=lead_section, page=page)


# --------------------------------------------------------------------------- #
# Article Polishing Module                                                     #
# --------------------------------------------------------------------------- #

class StormArticlePolishingModule(ArticlePolishingModule):
    """
    Finalises a draft article by adding a lead section and (optionally)
    removing redundant content.
    """

    def __init__(
        self,
        article_gen_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        article_polish_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
    ):
        self.article_gen_lm = article_gen_lm
        self.article_polish_lm = article_polish_lm

        self.polish_page = PolishPageModule(
            write_lead_engine=self.article_gen_lm,
            polish_engine=self.article_polish_lm,
        )

    def polish_article(
        self, topic: str, draft_article: StormArticle, remove_duplicate: bool = False
    ) -> StormArticle:
        """
        Polish the draft article.

        Args:
            topic: The article topic.
            draft_article: Output from the article-generation stage.
            remove_duplicate: When True, deduplicate the article body with an extra LM call.

        Returns:
            A polished StormArticle with an added lead/summary section.
        """
        article_text = draft_article.to_string()

        polish_result = self.polish_page(
            topic=topic,
            draft_page=article_text,
            polish_whole_page=remove_duplicate,
        )

        # Prepend the generated lead under a '# summary' heading.
        lead = f"# summary\n{polish_result.lead_section}"
        combined_text = "\n\n".join([lead, polish_result.page])

        polished_dict = ArticleTextProcessing.parse_article_into_dict(combined_text)
        polished_article = copy.deepcopy(draft_article)
        polished_article.insert_or_create_section(article_dict=polished_dict)
        polished_article.post_processing()

        return polished_article
