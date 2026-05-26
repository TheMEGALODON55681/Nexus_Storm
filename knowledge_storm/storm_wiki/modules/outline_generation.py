from typing import Union, Optional, Tuple

import dspy

from .callback import BaseCallbackHandler
from .storm_dataclass import StormInformationTable, StormArticle
from ...interface import OutlineGenerationModule
from ...utils import ArticleTextProcessing


# --------------------------------------------------------------------------- #
# DSPy Signatures                                                              #
# --------------------------------------------------------------------------- #

class WritePageOutline(dspy.Signature):
    """Draft a structured outline for a Wikipedia article.

    Formatting rules:
    1. Use '#' for top-level sections, '##' for subsections, '###' for sub-subsections, and so on.
    2. Include section headings only — no body text or explanatory notes.
    3. Do not include the topic name itself as a heading in the outline.
    """

    topic = dspy.InputField(prefix="Topic to outline: ", format=str)
    outline = dspy.OutputField(prefix="Wikipedia page outline:\n", format=str)


class WritePageOutlineFromConv(dspy.Signature):
    """Refine an existing Wikipedia outline using insights from a research conversation.

    A draft outline has already been created from prior knowledge. Improve it by
    incorporating specific details, angles, and gaps identified in the conversation below.

    Formatting rules:
    1. Use '#' for sections, '##' for subsections, '###' for sub-subsections, etc.
    2. Include headings only — no body text.
    3. Do not include the topic name itself as a heading.
    """

    topic = dspy.InputField(prefix="Topic to outline: ", format=str)
    conv = dspy.InputField(prefix="Research conversation:\n", format=str)
    old_outline = dspy.OutputField(prefix="Current draft outline:\n", format=str)
    outline = dspy.OutputField(
        prefix='Refined outline (use "#" for sections, "##" for subsections, ...):\n',
        format=str,
    )


# --------------------------------------------------------------------------- #
# DSPy Modules                                                                 #
# --------------------------------------------------------------------------- #

class NaiveOutlineGen(dspy.Module):
    """Generate an outline using only the LLM's parametric knowledge (no retrieval)."""

    def __init__(self):
        super().__init__()
        self.write_outline = dspy.Predict(WritePageOutline)

    def forward(self, topic: str):
        outline = self.write_outline(topic=topic).outline
        return dspy.Prediction(outline=outline)


class WriteOutline(dspy.Module):
    """Two-phase outline writer: draft from parametric knowledge, then refine from conversation."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.draft_outline = dspy.Predict(WritePageOutline)
        self.refine_outline = dspy.Predict(WritePageOutlineFromConv)
        self.engine = engine

    def forward(
        self,
        topic: str,
        dlg_history,
        old_outline: Optional[str] = None,
        callback_handler: BaseCallbackHandler = None,
    ):
        # Build a cleaned conversation string from the dialogue history.
        relevant_turns = [
            turn for turn in dlg_history
            if "topic you" not in turn.agent_utterance.lower()
            and "topic you" not in turn.user_utterance.lower()
        ]
        conv_text = "\n".join(
            f"Wikipedia Writer: {turn.user_utterance}\nExpert: {turn.agent_utterance}"
            for turn in relevant_turns
        )
        conv_text = ArticleTextProcessing.remove_citations(conv_text)
        conv_text = ArticleTextProcessing.limit_word_count_preserve_newline(conv_text, 5000)

        with dspy.settings.context(lm=self.engine):
            if old_outline is None:
                old_outline = ArticleTextProcessing.clean_up_outline(
                    self.draft_outline(topic=topic).outline
                )
                if callback_handler:
                    callback_handler.on_direct_outline_generation_end(outline=old_outline)

            refined = ArticleTextProcessing.clean_up_outline(
                self.refine_outline(
                    topic=topic, old_outline=old_outline, conv=conv_text
                ).outline
            )
            if callback_handler:
                callback_handler.on_outline_refinement_end(outline=refined)

        return dspy.Prediction(outline=refined, old_outline=old_outline)


# --------------------------------------------------------------------------- #
# Outline Generation Module                                                    #
# --------------------------------------------------------------------------- #

class StormOutlineGenerationModule(OutlineGenerationModule):
    """
    Orchestrates the outline-generation stage for the STORM pipeline.

    Aggregates all dialogue turns from the information table, runs the
    two-phase WriteOutline module, and returns structured StormArticle outlines.
    """

    def __init__(self, outline_gen_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.outline_gen_lm = outline_gen_lm
        self.write_outline = WriteOutline(engine=self.outline_gen_lm)

    def generate_outline(
        self,
        topic: str,
        information_table: StormInformationTable,
        old_outline: Optional[StormArticle] = None,
        callback_handler: BaseCallbackHandler = None,
        return_draft_outline: bool = False,
    ) -> Union[StormArticle, Tuple[StormArticle, StormArticle]]:
        """
        Generate a structured article outline from curated conversation data.

        Args:
            topic: The article topic.
            information_table: Curated information from the knowledge-curation stage.
            old_outline: An optional pre-existing outline to refine rather than replace.
            callback_handler: Optional handler for progress callbacks.
            return_draft_outline: When True, return both the final and draft outlines.

        Returns:
            Either a single StormArticle (final outline) or a tuple of
            (final_outline, draft_outline) when ``return_draft_outline`` is True.
        """
        if callback_handler is not None:
            callback_handler.on_information_organization_start()

        all_turns = sum(
            [conv for (_, conv) in information_table.conversations], []
        )

        result = self.write_outline(
            topic=topic,
            dlg_history=all_turns,
            callback_handler=callback_handler,
        )

        final_article = StormArticle.from_outline_str(
            topic=topic, outline_str=result.outline
        )
        draft_article = StormArticle.from_outline_str(
            topic=topic, outline_str=result.old_outline
        )

        if return_draft_outline:
            return final_article, draft_article
        return final_article
