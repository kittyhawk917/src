#!/usr/bin/env python3

import sys
import os
import re
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from anthropic import Anthropic


def llm_call(model: str, prompt: str, system_prompt: str = "") -> str:

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    messages = [{"role": "user", "content": prompt}]
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=messages,
        temperature=0.1,
    )
    return response.content[0].text


def extract_xml(text: str, tag: str) -> str:

    match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1) if match else ""


def generate_puzzle(instruction: str, model: str) -> str:

    prompt = f"""

    {instruction}

    Do not include explanations and thinking process in response.
    Return ONLY the tags, the final puzzle, and the final answer *strictly* in this format:

        <puzzle>
        # the created puzzle here
        </puzzle>

        <answer>
        # the answer of the created puzzle here
        </answer>

    """

    response = llm_call(model, prompt)
    return response


def main() -> int:

    # load environment variables from a specific file
    env_file = find_dotenv(Path(Path.home(), ".zshrc"))
    load_dotenv(env_file)

    # set environment variables
    os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

    # models
    generation_model = "claude-haiku-4-5"
    reflection_model = "claude-sonnet-4-5"


    # Generate initial code
    generation_instructions = f"""

        You are a creative puzzle maker. You are going to create a puzzle satisfying the following requirements:
        1. The puzzle has exactly one solution.
        2. The puzzle contains four groups of four items. Each group of four items share something in common.
        3. All four groups are different
        4. All sixteen group items are different.
        5. Categories will always be more specific than "5-LETTER-WORDS," "NAMES" or "VERBS."
        6. Use words that seem to belong to multiple categories.
        7. Here are examples of categories and items:
            "FISH": Bass, Flounder, Salmon, Trout
            "FIRE ___": Ant, Drill, Island, Opal
        
        Answer: List groups, something in common, and group items. Each group a line.
        Puzzle: Collect all group items. Shuffle and list them to be four lines of four items.
        
    """
    
    # use a fast model for generate initial puzzle
    response = generate_puzzle(generation_instructions, generation_model)

    puzzle = extract_xml(response, "puzzle")
    answer = extract_xml(response, "answer")

    print(f"""
          The initial puzzle:
          {puzzle}
          """)
    print(f"""
          The initial answer:
          {answer}
          """)


    # reflect and refine puzzle
    reflection_instructions = f"""

        Original groups: {answer}

        You are a critical puzzle maker. You are given four groups of four items. Each group of four items share something in common. 
        
        Your TASK is to replace the worst group in the original groups with a new group of four items. Use words that seem to belong to multiple categories.
        
        The updated puzzle must satisfy the following requirements:
        1. The puzzle has exactly one solution.
        2. All four groups are different.
        3. All sixteen group items are different.
        4. Here are examples of categories and items:
            "Bit of magic": CHARM, CURSE, HEX, SPELL
            "Places where things disappear": BERMUDA TRIANGLE, BLACK HOLE, COUCH CUSHIONS, DRYER
            "Associated with Philadelphia": BROTHERLY LOVE, CHEESESTEAK, LIBERTY BELL, ROCKY
            "Butter____": FINGERS, FLY, NUT, SCOTCH

        Retry the TASK if the requirements are not met.
        
        Answer: List group, something in common, and group items. Each group a line.
        Puzzle: Collect every items from all groups. Shuffle and list them to be four lines of four items.

    """

    # use a stronger reasoning model for reflection 
    response = generate_puzzle(reflection_instructions, reflection_model)

    puzzle = extract_xml(response, "puzzle")
    answer = extract_xml(response, "answer")

    print(f"""
          The refined puzzle:
          {puzzle}
          """)
    print(f"""
          The refined answer:
          {answer}
          """)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

