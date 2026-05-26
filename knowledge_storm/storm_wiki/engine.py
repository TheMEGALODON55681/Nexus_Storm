import json
import logging
import os
from dataclasses import dataclass, field
from typing import Union, Literal, Optional

import dspy

from .modules.article_generation import StormArticleGenerationModule
from .modules.article_polish import StormArticlePolishingModule
from .modules.callback import BaseCallbackHandler
from .modules.knowledge_curation import StormKnowledgeCurationModule
from .modules.outline_generation import StormOutlineGenerationModule
from .modules.persona_generator import StormPersonaGenerator
from .modules.storm_dataclass import StormInformationTable, StormArticle
from ..interface import Engine, LMConfigs, Retriever
from ..lm import LitellmModel
from ..utils import FileIOHelper, highlight_error, truncate_filename


class STORMWikiLMConfigs(LMConfigs):
    """Language-model configuration bundle for the STORM Wiki pipeline.

    Different pipeline stages have different complexity requirements, so each
    stage can be powered by a separate model to balance quality against cost.
    """

    def __init__(self):
        # Conversation simulation (excluding question-asking turns)
        self.conv_simulator_lm = None
        # Question generation during perspective-guided dialogue
        self.question_asker_lm = None
        # Outline generation from aggregated conversation
        self.outline_gen_lm = None
        # Section-level article writing
        self.article_gen_lm = None
        # Final article polishing / deduplication
        self.article_polish_lm = None

    def init_openai_model(
        self,
        openai_api_key: str,
        azure_api_key: str,
        openai_type: Literal["openai", "azure"],
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 0.9,
    ):
        """Convenience initialiser matching the original NAACL'24 paper setup."""
        shared_azure = {
            "api_key": azure_api_key,
            "temperature": temperature,
            "top_p": top_p,
            "api_base": api_base,
            "api_version": api_version,
        }
        shared_openai = {
            "api_key": openai_api_key,
            "temperature": temperature,
            "top_p": top_p,
            "api_base": None,
        }

        if openai_type == "openai":
            self.conv_simulator_lm = LitellmModel(
                model="gpt-4o-mini-2024-07-18", max_tokens=500, **shared_openai
            )
            self.question_asker_lm = LitellmModel(
                model="gpt-4o-mini-2024-07-18", max_tokens=500, **shared_openai
            )
            self.outline_gen_lm = LitellmModel(
                model="gpt-4-0125-preview", max_tokens=400, **shared_openai
            )
            self.article_gen_lm = LitellmModel(
                model="gpt-4o-2024-05-13", max_tokens=700, **shared_openai
            )
            self.article_polish_lm = LitellmModel(
                model="gpt-4o-2024-05-13", max_tokens=4000, **shared_openai
            )
        elif openai_type == "azure":
            self.conv_simulator_lm = LitellmModel(
                model="azure/gpt-4o-mini-2024-07-18", max_tokens=500, **shared_openai
            )
            self.question_asker_lm = LitellmModel(
                model="azure/gpt-4o-mini-2024-07-18",
                max_tokens=500,
                **shared_azure,
                model_type="chat",
            )
            self.outline_gen_lm = LitellmModel(
                model="azure/gpt-4o", max_tokens=400, **shared_azure, model_type="chat"
            )
            self.article_gen_lm = LitellmModel(
                model="azure/gpt-4o-mini-2024-07-18",
                max_tokens=700,
                **shared_azure,
                model_type="chat",
            )
            self.article_polish_lm = LitellmModel(
                model="azure/gpt-4o-mini-2024-07-18",
                max_tokens=4000,
                **shared_azure,
                model_type="chat",
            )
        else:
            logging.warning(
                "Unrecognised openai_type '%s'. Default LM config not applied.",
                openai_type,
            )

    # ------------------------------------------------------------------ #
    # Individual setters                                                   #
    # ------------------------------------------------------------------ #

    def set_conv_simulator_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.conv_simulator_lm = model

    def set_question_asker_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.question_asker_lm = model

    def set_outline_gen_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.outline_gen_lm = model

    def set_article_gen_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.article_gen_lm = model

    def set_article_polish_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.article_polish_lm = model


@dataclass
class STORMWikiRunnerArguments:
    """Runtime configuration for the STORM Wiki pipeline."""

    output_dir: str = field(
        metadata={"help": "Directory where pipeline outputs are written."},
    )
    max_conv_turn: int = field(
        default=3,
        metadata={"help": "Maximum dialogue turns per simulated conversation."},
    )
    max_perspective: int = field(
        default=3,
        metadata={"help": "Maximum distinct writer perspectives to simulate."},
    )
    max_search_queries_per_turn: int = field(
        default=3,
        metadata={"help": "Maximum search queries issued per dialogue turn."},
    )
    disable_perspective: bool = field(
        default=False,
        metadata={"help": "When True, skip perspective-guided question asking."},
    )
    search_top_k: int = field(
        default=3,
        metadata={"help": "Number of top search results kept per query."},
    )
    retrieve_top_k: int = field(
        default=3,
        metadata={"help": "Number of references retrieved per article section."},
    )
    max_thread_num: int = field(
        default=10,
        metadata={
            "help": (
                "Thread pool size for parallel operations. "
                "Reduce if you encounter rate-limit errors from the LM API."
            )
        },
    )


class STORMWikiRunner(Engine):
    """Orchestrates the full STORM Wiki article-generation pipeline."""

    def __init__(
        self, args: STORMWikiRunnerArguments, lm_configs: STORMWikiLMConfigs, rm
    ):
        super().__init__(lm_configs=lm_configs)
        self.args = args
        self.lm_configs = lm_configs

        self.retriever = Retriever(rm=rm, max_thread=self.args.max_thread_num)

        persona_generator = StormPersonaGenerator(self.lm_configs.question_asker_lm)

        self.storm_knowledge_curation_module = StormKnowledgeCurationModule(
            retriever=self.retriever,
            persona_generator=persona_generator,
            conv_simulator_lm=self.lm_configs.conv_simulator_lm,
            question_asker_lm=self.lm_configs.question_asker_lm,
            max_search_queries_per_turn=self.args.max_search_queries_per_turn,
            search_top_k=self.args.search_top_k,
            max_conv_turn=self.args.max_conv_turn,
            max_thread_num=self.args.max_thread_num,
        )
        self.storm_outline_generation_module = StormOutlineGenerationModule(
            outline_gen_lm=self.lm_configs.outline_gen_lm
        )
        self.storm_article_generation = StormArticleGenerationModule(
            article_gen_lm=self.lm_configs.article_gen_lm,
            retrieve_top_k=self.args.retrieve_top_k,
            max_thread_num=self.args.max_thread_num,
        )
        self.storm_article_polishing_module = StormArticlePolishingModule(
            article_gen_lm=self.lm_configs.article_gen_lm,
            article_polish_lm=self.lm_configs.article_polish_lm,
        )

        self.lm_configs.init_check()
        self.apply_decorators()

    # ------------------------------------------------------------------ #
    # Pipeline stage runners                                               #
    # ------------------------------------------------------------------ #

    def run_knowledge_curation_module(
        self,
        ground_truth_url: str = "None",
        callback_handler: BaseCallbackHandler = None,
    ) -> StormInformationTable:
        information_table, conversation_log = (
            self.storm_knowledge_curation_module.research(
                topic=self.topic,
                ground_truth_url=ground_truth_url,
                callback_handler=callback_handler,
                max_perspective=self.args.max_perspective,
                disable_perspective=False,
                return_conversation_log=True,
            )
        )

        FileIOHelper.dump_json(
            conversation_log,
            os.path.join(self.article_output_dir, "conversation_log.json"),
        )
        information_table.dump_url_to_info(
            os.path.join(self.article_output_dir, "raw_search_results.json")
        )
        return information_table

    def run_outline_generation_module(
        self,
        information_table: StormInformationTable,
        callback_handler: BaseCallbackHandler = None,
    ) -> StormArticle:
        outline, draft_outline = self.storm_outline_generation_module.generate_outline(
            topic=self.topic,
            information_table=information_table,
            return_draft_outline=True,
            callback_handler=callback_handler,
        )
        outline.dump_outline_to_file(
            os.path.join(self.article_output_dir, "storm_gen_outline.txt")
        )
        draft_outline.dump_outline_to_file(
            os.path.join(self.article_output_dir, "direct_gen_outline.txt")
        )
        return outline

    def run_article_generation_module(
        self,
        outline: StormArticle,
        information_table=StormInformationTable,
        callback_handler: BaseCallbackHandler = None,
    ) -> StormArticle:
        draft_article = self.storm_article_generation.generate_article(
            topic=self.topic,
            information_table=information_table,
            article_with_outline=outline,
            callback_handler=callback_handler,
        )
        draft_article.dump_article_as_plain_text(
            os.path.join(self.article_output_dir, "storm_gen_article.txt")
        )
        draft_article.dump_reference_to_file(
            os.path.join(self.article_output_dir, "url_to_info.json")
        )
        return draft_article

    def run_article_polishing_module(
        self, draft_article: StormArticle, remove_duplicate: bool = False
    ) -> StormArticle:
        polished_article = self.storm_article_polishing_module.polish_article(
            topic=self.topic,
            draft_article=draft_article,
            remove_duplicate=remove_duplicate,
        )
        FileIOHelper.write_str(
            polished_article.to_string(),
            os.path.join(self.article_output_dir, "storm_gen_article_polished.txt"),
        )
        return polished_article

    def post_run(self):
        """Persist the run configuration and LLM call history to disk."""
        config_log = self.lm_configs.log()
        FileIOHelper.dump_json(
            config_log, os.path.join(self.article_output_dir, "run_config.json")
        )

        llm_call_history = self.lm_configs.collect_and_reset_lm_history()
        history_path = os.path.join(self.article_output_dir, "llm_call_history.jsonl")
        with open(history_path, "w") as fh:
            for call in llm_call_history:
                # kwargs are already captured in run_config.json; skip here to avoid duplication.
                call.pop("kwargs", None)
                fh.write(json.dumps(call) + "\n")

    # ------------------------------------------------------------------ #
    # Local filesystem loaders                                             #
    # ------------------------------------------------------------------ #

    def _load_information_table_from_local_fs(self, path):
        assert os.path.exists(path), highlight_error(
            f"{path} not found. Run with --do-research to generate conversation_log.json first."
        )
        return StormInformationTable.from_conversation_log_file(path)

    def _load_outline_from_local_fs(self, topic, path):
        assert os.path.exists(path), highlight_error(
            f"{path} not found. Run with --do-generate-outline to generate storm_gen_outline.txt first."
        )
        return StormArticle.from_outline_file(topic=topic, file_path=path)

    def _load_draft_article_from_local_fs(self, topic, draft_path, url_info_path):
        assert os.path.exists(draft_path), highlight_error(
            f"{draft_path} not found. Run with --do-generate-article first."
        )
        assert os.path.exists(url_info_path), highlight_error(
            f"{url_info_path} not found. Run with --do-generate-article first."
        )
        article_text = FileIOHelper.load_str(draft_path)
        references = FileIOHelper.load_json(url_info_path)
        return StormArticle.from_string(
            topic_name=topic, article_text=article_text, references=references
        )

    # ------------------------------------------------------------------ #
    # Main entry point                                                     #
    # ------------------------------------------------------------------ #

    def run(
        self,
        topic: str,
        ground_truth_url: str = "",
        do_research: bool = True,
        do_generate_outline: bool = True,
        do_generate_article: bool = True,
        do_polish_article: bool = True,
        remove_duplicate: bool = False,
        callback_handler: BaseCallbackHandler = BaseCallbackHandler(),
    ):
        """
        Execute the STORM pipeline end-to-end (or any subset of stages).

        Args:
            topic: Research topic in natural language.
            ground_truth_url: URL of an authoritative article to exclude from search.
            do_research: Run the knowledge-curation stage.
            do_generate_outline: Run the outline-generation stage.
            do_generate_article: Run the article-generation stage.
            do_polish_article: Run the article-polishing stage.
            remove_duplicate: Pass True to deduplicate content during polishing.
            callback_handler: Optional handler for intermediate stage callbacks.
        """
        assert any(
            [do_research, do_generate_outline, do_generate_article, do_polish_article]
        ), highlight_error(
            "No pipeline stage selected. Specify at least one of: "
            "--do-research, --do-generate-outline, --do-generate-article, --do-polish-article"
        )

        self.topic = topic
        self.article_dir_name = truncate_filename(
            topic.replace(" ", "_").replace("/", "_")
        )
        self.article_output_dir = os.path.join(
            self.args.output_dir, self.article_dir_name
        )
        os.makedirs(self.article_output_dir, exist_ok=True)

        # Stage 1: Knowledge curation
        information_table: StormInformationTable = None
        if do_research:
            information_table = self.run_knowledge_curation_module(
                ground_truth_url=ground_truth_url, callback_handler=callback_handler
            )

        # Stage 2: Outline generation
        outline: StormArticle = None
        if do_generate_outline:
            if information_table is None:
                information_table = self._load_information_table_from_local_fs(
                    os.path.join(self.article_output_dir, "conversation_log.json")
                )
            outline = self.run_outline_generation_module(
                information_table=information_table, callback_handler=callback_handler
            )

        # Stage 3: Article generation
        draft_article: StormArticle = None
        if do_generate_article:
            if information_table is None:
                information_table = self._load_information_table_from_local_fs(
                    os.path.join(self.article_output_dir, "conversation_log.json")
                )
            if outline is None:
                outline = self._load_outline_from_local_fs(
                    topic=topic,
                    path=os.path.join(self.article_output_dir, "storm_gen_outline.txt"),
                )
            draft_article = self.run_article_generation_module(
                outline=outline,
                information_table=information_table,
                callback_handler=callback_handler,
            )

        # Stage 4: Article polishing
        if do_polish_article:
            if draft_article is None:
                draft_article = self._load_draft_article_from_local_fs(
                    topic=topic,
                    draft_path=os.path.join(self.article_output_dir, "storm_gen_article.txt"),
                    url_info_path=os.path.join(self.article_output_dir, "url_to_info.json"),
                )
            self.run_article_polishing_module(
                draft_article=draft_article, remove_duplicate=remove_duplicate
            )
