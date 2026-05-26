import logging
import re
from typing import Union, List

import dspy
import requests
from bs4 import BeautifulSoup


def get_wiki_page_title_and_toc(url: str):
    """Fetch the title and table of contents from a Wikipedia page URL.

    Args:
        url: Full URL of the Wikipedia article.

    Returns:
        Tuple of (main_title, toc_string).
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    main_title = (
        soup.find("h1").text.replace("[edit]", "").strip().replace("\xa0", " ")
    )

    _EXCLUDED_SECTIONS = {"Contents", "See also", "Notes", "References", "External links"}

    toc_lines = []
    indent_stack = []

    for header in soup.find_all(["h2", "h3", "h4", "h5", "h6"]):
        depth = int(header.name[1])
        section_title = header.text.replace("[edit]", "").strip().replace("\xa0", " ")

        if section_title in _EXCLUDED_SECTIONS:
            continue

        while indent_stack and depth <= indent_stack[-1]:
            indent_stack.pop()
        indent_stack.append(depth)

        indent = "  " * (len(indent_stack) - 1)
        toc_lines.append(f"{indent}{section_title}")

    return main_title, "\n".join(toc_lines)


# --------------------------------------------------------------------------- #
# DSPy Signatures                                                              #
# --------------------------------------------------------------------------- #

class FindRelatedTopic(dspy.Signature):
    """Identify Wikipedia pages on subjects closely related to the given topic.

    These related pages will be used to understand the typical structure and coverage
    of similar Wikipedia articles. List each URL on a separate line.
    """

    topic = dspy.InputField(prefix="Topic of interest:", format=str)
    related_topics = dspy.OutputField(format=str)


class GenPersona(dspy.Signature):
    """Assemble a diverse editorial team for writing a comprehensive Wikipedia article.

    Each editor should represent a distinct perspective, background, or area of focus
    relevant to the topic. Draw inspiration from the related Wikipedia pages provided.

    Format your response as a numbered list:
    1. <short role label>: <description of editorial focus>
    2. <short role label>: <description of editorial focus>
    ...
    """

    topic = dspy.InputField(prefix="Topic of interest:", format=str)
    examples = dspy.InputField(
        prefix="Outlines of related Wikipedia pages for reference:\n", format=str
    )
    personas = dspy.OutputField(format=str)


# --------------------------------------------------------------------------- #
# DSPy Modules                                                                 #
# --------------------------------------------------------------------------- #

class CreateWriterWithPersona(dspy.Module):
    """Generates a set of writer personas by analysing related Wikipedia pages."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.find_related_topic = dspy.ChainOfThought(FindRelatedTopic)
        self.gen_persona = dspy.ChainOfThought(GenPersona)
        self.engine = engine

    def forward(self, topic: str, draft=None):
        with dspy.settings.context(lm=self.engine):
            related_topics_output = self.find_related_topic(topic=topic).related_topics

            # Extract URLs from the free-text output.
            urls = [
                line[line.find("http"):]
                for line in related_topics_output.split("\n")
                if "http" in line
            ]

            reference_outlines = []
            for url in urls:
                try:
                    title, toc = get_wiki_page_title_and_toc(url)
                    reference_outlines.append(f"Title: {title}\nTable of Contents: {toc}")
                except Exception as exc:
                    logging.error("Failed to retrieve TOC from %s: %s", url, exc)

            if not reference_outlines:
                reference_outlines.append("N/A")

            raw_personas_output = self.gen_persona(
                topic=topic, examples="\n----------\n".join(reference_outlines)
            ).personas

        personas = []
        for line in raw_personas_output.split("\n"):
            match = re.search(r"\d+\.\s*(.*)", line)
            if match:
                personas.append(match.group(1))

        return dspy.Prediction(
            personas=personas,
            raw_personas_output=raw_personas_output,
            related_topics=related_topics_output,
        )


# --------------------------------------------------------------------------- #
# Public Generator                                                             #
# --------------------------------------------------------------------------- #

class StormPersonaGenerator:
    """
    Generates a list of writer personas for a given research topic.

    A default "Basic fact writer" persona is always prepended to the returned list
    to ensure broad factual coverage alongside the specialised perspectives.

    Args:
        engine: The language model used to generate personas.
    """

    _DEFAULT_PERSONA = (
        "Basic fact writer: "
        "Focuses on broadly covering the fundamental facts about the topic."
    )

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.persona_creator = CreateWriterWithPersona(engine=engine)

    def generate_persona(self, topic: str, max_num_persona: int = 3) -> List[str]:
        """
        Produce a persona list for the given topic.

        Args:
            topic: The research topic.
            max_num_persona: Maximum number of specialised personas to include
                (excluding the default basic-fact persona).

        Returns:
            List of persona description strings. The first entry is always the
            default basic-fact persona.
        """
        generation_result = self.persona_creator(topic=topic)
        specialised_personas = generation_result.personas[:max_num_persona]
        return [self._DEFAULT_PERSONA] + specialised_personas
