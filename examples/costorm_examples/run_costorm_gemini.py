"""
Co-STORM pipeline powered by Google Gemini and DuckDuckGo search engine.
You need to set up the following environment variables to run this script:
    - GEMINI_API_KEY: Your Google Gemini API key (free at aistudio.google.com)

No search engine API key needed — DuckDuckGo is used for free web search.

Output will be structured as below:
args.output_dir/
    report.md          # Final article generated
    instance_dump.json # Full instance data
    log.json           # Log of information-seeking conversation
"""

import os
import json
from argparse import ArgumentParser
from knowledge_storm.collaborative_storm.engine import (
    CollaborativeStormLMConfigs,
    RunnerArgument,
    CoStormRunner,
)
from knowledge_storm.collaborative_storm.modules.callback import (
    LocalConsolePrintCallBackHandler,
)
from knowledge_storm.lm import LitellmModel
from knowledge_storm.logging_wrapper import LoggingWrapper
from knowledge_storm.rm import DuckDuckGoSearchRM


def main(args):
    # Set your Gemini API key
    os.environ["GEMINI_API_KEY"] = "your_gemini_api_key_here"  # 👈 Replace this with your actual key

    lm_config: CollaborativeStormLMConfigs = CollaborativeStormLMConfigs()

    gemini_kwargs = {
        "api_key": os.getenv("GEMINI_API_KEY"),
        "temperature": 1.0,
        "top_p": 0.9,
    }

    # Using Gemini 1.5 Flash — fast and free tier available
    gemini_model_name = "gemini/gemini-1.5-flash"

    question_answering_lm = LitellmModel(model=gemini_model_name, max_tokens=1000, **gemini_kwargs)
    discourse_manage_lm = LitellmModel(model=gemini_model_name, max_tokens=500, **gemini_kwargs)
    utterance_polishing_lm = LitellmModel(model=gemini_model_name, max_tokens=2000, **gemini_kwargs)
    warmstart_outline_gen_lm = LitellmModel(model=gemini_model_name, max_tokens=500, **gemini_kwargs)
    question_asking_lm = LitellmModel(model=gemini_model_name, max_tokens=300, **gemini_kwargs)
    knowledge_base_lm = LitellmModel(model=gemini_model_name, max_tokens=1000, **gemini_kwargs)

    lm_config.set_question_answering_lm(question_answering_lm)
    lm_config.set_discourse_manage_lm(discourse_manage_lm)
    lm_config.set_utterance_polishing_lm(utterance_polishing_lm)
    lm_config.set_warmstart_outline_gen_lm(warmstart_outline_gen_lm)
    lm_config.set_question_asking_lm(question_asking_lm)
    lm_config.set_knowledge_base_lm(knowledge_base_lm)

    topic = input("Topic: ")
    runner_argument = RunnerArgument(
        topic=topic,
        retrieve_top_k=args.retrieve_top_k,
        max_search_queries=args.max_search_queries,
        total_conv_turn=args.total_conv_turn,
        max_search_thread=args.max_search_thread,
        max_search_queries_per_turn=args.max_search_queries_per_turn,
        warmstart_max_num_experts=args.warmstart_max_num_experts,
        warmstart_max_turn_per_experts=args.warmstart_max_turn_per_experts,
        warmstart_max_thread=args.warmstart_max_thread,
        max_thread_num=args.max_thread_num,
        max_num_round_table_experts=args.max_num_round_table_experts,
        moderator_override_N_consecutive_answering_turn=args.moderator_override_N_consecutive_answering_turn,
        node_expansion_trigger_count=args.node_expansion_trigger_count,
    )

    logging_wrapper = LoggingWrapper(lm_config)
    callback_handler = (
        LocalConsolePrintCallBackHandler() if args.enable_log_print else None
    )

    # Using DuckDuckGo — no API key needed!
    rm = DuckDuckGoSearchRM(
        k=runner_argument.retrieve_top_k, safe_search="On", region="us-en"
    )

    costorm_runner = CoStormRunner(
        lm_config=lm_config,
        runner_argument=runner_argument,
        logging_wrapper=logging_wrapper,
        rm=rm,
        callback_handler=callback_handler,
    )

    # Warm start the system
    costorm_runner.warm_start()

    # Observe Co-STORM LLM agent utterance for 1 turn
    for _ in range(1):
        conv_turn = costorm_runner.step()
        print(f"**{conv_turn.role}**: {conv_turn.utterance}\n")

    # Actively engage by injecting your own utterance
    your_utterance = input("Your utterance: ")
    costorm_runner.step(user_utterance=your_utterance)

    # Continue observing
    conv_turn = costorm_runner.step()
    print(f"**{conv_turn.role}**: {conv_turn.utterance}\n")

    # Generate report
    costorm_runner.knowledge_base.reorganize()
    article = costorm_runner.generate_report()

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "report.md"), "w") as f:
        f.write(article)

    instance_copy = costorm_runner.to_dict()
    with open(os.path.join(args.output_dir, "instance_dump.json"), "w") as f:
        json.dump(instance_copy, f, indent=2)

    log_dump = costorm_runner.dump_logging_and_reset()
    with open(os.path.join(args.output_dir, "log.json"), "w") as f:
        json.dump(log_dump, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="./results/co-storm", help="Directory to store the outputs.")
    parser.add_argument("--retrieve_top_k", type=int, default=10)
    parser.add_argument("--max_search_queries", type=int, default=2)
    parser.add_argument("--total_conv_turn", type=int, default=20)
    parser.add_argument("--max_search_thread", type=int, default=5)
    parser.add_argument("--max_search_queries_per_turn", type=int, default=3)
    parser.add_argument("--warmstart_max_num_experts", type=int, default=3)
    parser.add_argument("--warmstart_max_turn_per_experts", type=int, default=2)
    parser.add_argument("--warmstart_max_thread", type=int, default=3)
    parser.add_argument("--max_thread_num", type=int, default=10)
    parser.add_argument("--max_num_round_table_experts", type=int, default=2)
    parser.add_argument("--moderator_override_N_consecutive_answering_turn", type=int, default=3)
    parser.add_argument("--node_expansion_trigger_count", type=int, default=10)
    parser.add_argument("--enable_log_print", action="store_true")

    main(parser.parse_args())
