import concurrent.futures
import logging
import os
from concurrent.futures import as_completed
from typing import Union, List, Tuple, Optional, Dict

import dspy

from .callback import BaseCallbackHandler
from .persona_generator import StormPersonaGenerator
from .storm_dataclass import DialogueTurn, StormInformationTable
from ...interface import KnowledgeCurationModule, Retriever, Information
from ...utils import ArticleTextProcessing

try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx
    streamlit_connection = True
except ImportError:
    streamlit_connection = False

script_dir = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# DSPy Signatures                                                              #
# --------------------------------------------------------------------------- #

class AskQuestion(dspy.Signature):
    """You are an experienced Wikipedia writer chatting with a subject-matter expert
    to gather information for the article you are writing. Ask focused, relevant questions
    one at a time. When you have no further questions, say exactly:
    \"Thank you so much for your help!\" to close the conversation.
    Do not repeat questions you have already asked.
    """

    topic = dspy.InputField(prefix="Topic you want to write: ", format=str)
    conv = dspy.InputField(prefix="Conversation history:\n", format=str)
    question = dspy.OutputField(format=str)


class AskQuestionWithPersona(dspy.Signature):
    """You are an experienced Wikipedia writer with a specific editorial focus.
    Chat with the expert to gather the information you need for your article.
    Ask focused, relevant questions one at a time. When you have no further questions,
    say exactly: \"Thank you so much for your help!\" to end the conversation.
    Do not repeat questions you have already asked.
    """

    topic = dspy.InputField(prefix="Topic you want to write: ", format=str)
    persona = dspy.InputField(
        prefix="Your editorial focus beyond being a Wikipedia writer: ", format=str
    )
    conv = dspy.InputField(prefix="Conversation history:\n", format=str)
    question = dspy.OutputField(format=str)


class QuestionToQuery(dspy.Signature):
    """Convert a research question into web search queries.
    Output each query on its own line, prefixed with a dash, like this:
    - query 1
    - query 2
    ...
    - query n
    """

    topic = dspy.InputField(prefix="Topic under discussion: ", format=str)
    question = dspy.InputField(prefix="Question to answer: ", format=str)
    queries = dspy.OutputField(format=str)


class AnswerQuestion(dspy.Signature):
    """You are a knowledgeable expert helping a Wikipedia writer research a topic.
    Using only the gathered information below, compose a thorough, well-supported response.
    Every claim should be traceable to the provided sources. If the gathered information
    is insufficient, say so clearly and explain what is missing rather than speculating.
    """

    topic = dspy.InputField(prefix="Topic under discussion:", format=str)
    conv = dspy.InputField(prefix="Question:\n", format=str)
    info = dspy.InputField(prefix="Gathered information:\n", format=str)
    answer = dspy.OutputField(
        prefix="Compose your response, citing as many distinct sources as possible. Do not hallucinate.\n",
        format=str,
    )


# --------------------------------------------------------------------------- #
# DSPy Modules                                                                 #
# --------------------------------------------------------------------------- #

class WikiWriter(dspy.Module):
    """Generates the next question from the Wikipedia writer's perspective.

    Supports both generic and persona-guided question asking.
    """

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.ask_question_with_persona = dspy.ChainOfThought(AskQuestionWithPersona)
        self.ask_question = dspy.ChainOfThought(AskQuestion)
        self.engine = engine

    def forward(
        self,
        topic: str,
        persona: str,
        dialogue_turns: List[DialogueTurn],
        draft_page=None,
    ):
        # Build a trimmed conversation string to stay within context limits.
        conv_lines = []
        for turn in dialogue_turns[:-4]:
            conv_lines.append(
                f"You: {turn.user_utterance}\nExpert: [omitted for brevity]"
            )
        for turn in dialogue_turns[-4:]:
            conv_lines.append(
                f"You: {turn.user_utterance}\n"
                f"Expert: {ArticleTextProcessing.remove_citations(turn.agent_utterance)}"
            )

        conv = "\n".join(conv_lines).strip() or "N/A"
        conv = ArticleTextProcessing.limit_word_count_preserve_newline(conv, 2500)

        with dspy.settings.context(lm=self.engine):
            if persona and persona.strip():
                question = self.ask_question_with_persona(
                    topic=topic, persona=persona, conv=conv
                ).question
            else:
                question = self.ask_question(
                    topic=topic, persona=persona, conv=conv
                ).question

        return dspy.Prediction(question=question)


class TopicExpert(dspy.Module):
    """Answers writer questions using retrieval-augmented generation.

    Steps:
    1. Decompose the question into search queries.
    2. Retrieve relevant documents.
    3. Synthesise a grounded answer from the retrieved snippets.
    """

    def __init__(
        self,
        engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        max_search_queries: int,
        search_top_k: int,
        retriever: Retriever,
    ):
        super().__init__()
        self.generate_queries = dspy.Predict(QuestionToQuery)
        self.retriever = retriever
        self.answer_question = dspy.Predict(AnswerQuestion)
        self.engine = engine
        self.max_search_queries = max_search_queries
        self.search_top_k = search_top_k

    def forward(self, topic: str, question: str, ground_truth_url: str):
        with dspy.settings.context(lm=self.engine, show_guidelines=False):
            raw_queries = self.generate_queries(topic=topic, question=question).queries
            # Parse bullet-list output into individual query strings.
            parsed_queries = [
                q.replace("-", "").strip().strip('"').strip("\u201c").strip()
                for q in raw_queries.split("\n")
            ]
            search_queries = parsed_queries[: self.max_search_queries]

            retrieved: List[Information] = self.retriever.retrieve(
                list(set(search_queries)), exclude_urls=[ground_truth_url]
            )

            if retrieved:
                info_text = ""
                for idx, source in enumerate(retrieved):
                    info_text += "\n".join(
                        f"[{idx + 1}]: {snippet}" for snippet in source.snippets[:1]
                    )
                    info_text += "\n\n"

                info_text = ArticleTextProcessing.limit_word_count_preserve_newline(
                    info_text, 1000
                )

                try:
                    answer = self.answer_question(
                        topic=topic, conv=question, info=info_text
                    ).answer
                    answer = ArticleTextProcessing.remove_uncompleted_sentences_with_citations(
                        answer
                    )
                except Exception as exc:
                    logging.error("Answer generation failed: %s", exc)
                    answer = "I cannot answer this question at the moment. Please try a different question."
            else:
                answer = "No relevant information found for this question. Please try a different question."

        return dspy.Prediction(
            queries=search_queries, searched_results=retrieved, answer=answer
        )


class ConvSimulator(dspy.Module):
    """Simulates a multi-turn dialogue between a Wikipedia writer and a topic expert."""

    def __init__(
        self,
        topic_expert_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        question_asker_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        retriever: Retriever,
        max_search_queries_per_turn: int,
        search_top_k: int,
        max_turn: int,
    ):
        super().__init__()
        self.wiki_writer = WikiWriter(engine=question_asker_engine)
        self.topic_expert = TopicExpert(
            engine=topic_expert_engine,
            max_search_queries=max_search_queries_per_turn,
            search_top_k=search_top_k,
            retriever=retriever,
        )
        self.max_turn = max_turn

    def forward(
        self,
        topic: str,
        persona: str,
        ground_truth_url: str,
        callback_handler: BaseCallbackHandler,
    ):
        """
        Args:
            topic: The article topic.
            persona: Writer persona guiding the question focus.
            ground_truth_url: Excluded from retrieval to prevent evaluation leakage.
            callback_handler: Receives callbacks after each completed turn.
        """
        dialogue_history: List[DialogueTurn] = []

        for _ in range(self.max_turn):
            writer_question = self.wiki_writer(
                topic=topic, persona=persona, dialogue_turns=dialogue_history
            ).question

            if not writer_question:
                logging.error("Writer generated an empty question; stopping.")
                break
            if writer_question.startswith("Thank you so much for your help!"):
                break

            expert_output = self.topic_expert(
                topic=topic, question=writer_question, ground_truth_url=ground_truth_url
            )

            turn = DialogueTurn(
                agent_utterance=expert_output.answer,
                user_utterance=writer_question,
                search_queries=expert_output.queries,
                search_results=expert_output.searched_results,
            )
            dialogue_history.append(turn)
            callback_handler.on_dialogue_turn_end(dlg_turn=turn)

        return dspy.Prediction(dlg_history=dialogue_history)


# --------------------------------------------------------------------------- #
# Knowledge Curation Module                                                    #
# --------------------------------------------------------------------------- #

class StormKnowledgeCurationModule(KnowledgeCurationModule):
    """
    Drives the knowledge-curation stage of the STORM pipeline.

    Runs one simulated conversation per writer persona in parallel,
    then aggregates the collected information into a StormInformationTable.
    """

    def __init__(
        self,
        retriever: Retriever,
        persona_generator: Optional[StormPersonaGenerator],
        conv_simulator_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        question_asker_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        max_search_queries_per_turn: int,
        search_top_k: int,
        max_conv_turn: int,
        max_thread_num: int,
    ):
        self.retriever = retriever
        self.persona_generator = persona_generator
        self.conv_simulator_lm = conv_simulator_lm
        self.search_top_k = search_top_k
        self.max_thread_num = max_thread_num

        self.conv_simulator = ConvSimulator(
            topic_expert_engine=conv_simulator_lm,
            question_asker_engine=question_asker_lm,
            retriever=retriever,
            max_search_queries_per_turn=max_search_queries_per_turn,
            search_top_k=search_top_k,
            max_turn=max_conv_turn,
        )

    def _get_considered_personas(self, topic: str, max_num_persona: int) -> List[str]:
        return self.persona_generator.generate_persona(
            topic=topic, max_num_persona=max_num_persona
        )

    def _run_conversations(
        self,
        conv_simulator,
        topic: str,
        ground_truth_url: str,
        personas: List[str],
        callback_handler: BaseCallbackHandler,
    ) -> List[Tuple[str, List[DialogueTurn]]]:
        """
        Execute one conversation per persona concurrently.

        Returns:
            List of (persona, cleaned_dialogue_history) tuples.
        """

        def run_single(persona):
            return conv_simulator(
                topic=topic,
                ground_truth_url=ground_truth_url,
                persona=persona,
                callback_handler=callback_handler,
            )

        num_workers = min(self.max_thread_num, len(personas))
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_map = {
                executor.submit(run_single, persona): persona for persona in personas
            }

            if streamlit_connection:
                for thread in executor._threads:
                    add_script_run_ctx(thread)

            for future in as_completed(future_map):
                persona = future_map[future]
                conv_result = future.result()
                cleaned = ArticleTextProcessing.clean_up_citation(conv_result).dlg_history
                results.append((persona, cleaned))

        return results

    def research(
        self,
        topic: str,
        ground_truth_url: str,
        callback_handler: BaseCallbackHandler,
        max_perspective: int = 0,
        disable_perspective: bool = True,
        return_conversation_log: bool = False,
    ) -> Union[StormInformationTable, Tuple[StormInformationTable, Dict]]:
        """
        Run the knowledge-curation stage for a given topic.

        Args:
            topic: Research topic in natural language.
            ground_truth_url: URL to exclude from retrieval.
            callback_handler: Receives stage-level progress callbacks.
            max_perspective: Upper bound on simulated perspectives.
            disable_perspective: When True, use a single generic persona.
            return_conversation_log: When True, also return the raw log dict.

        Returns:
            A StormInformationTable, or a (table, log) tuple when
            ``return_conversation_log`` is True.
        """
        callback_handler.on_identify_perspective_start()

        if disable_perspective:
            personas = [""]
        else:
            personas = self._get_considered_personas(
                topic=topic, max_num_persona=max_perspective
            )

        callback_handler.on_identify_perspective_end(perspectives=personas)

        callback_handler.on_information_gathering_start()
        conversations = self._run_conversations(
            conv_simulator=self.conv_simulator,
            topic=topic,
            ground_truth_url=ground_truth_url,
            personas=personas,
            callback_handler=callback_handler,
        )

        information_table = StormInformationTable(conversations)
        callback_handler.on_information_gathering_end()

        if return_conversation_log:
            return information_table, StormInformationTable.construct_log_dict(conversations)
        return information_table
