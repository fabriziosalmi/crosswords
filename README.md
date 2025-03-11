# Interactive Crossword Puzzle Generator

This Python script (`crossword.py`) generates interactive crossword puzzles in HTML format, leveraging Language Models (LLMs) for clue generation. 

## Introduction

It offers a high degree of customization, including grid size, black square ratio, word lists, and LLM parameters.  The script uses a combination of constraint satisfaction, backtracking search, and (optionally) parallel processing to efficiently create crossword puzzles. 

## Features

*   **Customizable Grid:** Control the width, height, and percentage of black squares in the grid.  Supports both random grid generation and manual grid input (via string or file).
*   **LLM-Powered Clues:** Generates clues using a specified LLM (defaults to a locally served Mistral model via LM Studio).  Includes robust filtering and retries for high-quality clues.
*   **Interactive HTML Output:** Produces a self-contained, interactive HTML file where users can directly input answers and check their solutions.
*   **Word List Support:** Uses a provided word list (one word per line) to populate the crossword.  Calculates word frequencies for better word selection.
*   **Backtracking Search:** Employs a sophisticated backtracking algorithm with constraint propagation (forward checking) to ensure puzzle solvability.
*   **Parallel Processing (Optional):** Uses `ThreadPoolExecutor` to parallelize word placement attempts, significantly speeding up generation for larger grids.
*   **Difficulty Levels:** Introduces difficulty levels (easy, medium, hard) that affect word selection by prioritizing common or rare words.
*   **Beam Search:** Limits the branching factor during the search process, improving performance.
*   **Comprehensive Logging and Statistics:**  Provides detailed logging and statistics about the generation process, including attempts, backtracks, and time spent.
*   **Caching:** Caches word placement validity and generated definitions to reduce redundant calculations and LLM calls.
*   **Rich CLI Output:** Uses the `rich` library for visually appealing console output, including progress bars and grid previews.

## Requirements

*   Python 3.7+
*   `langchain-core`
*   `langchain-openai`
*   `nltk`
*   `rich`

Install the dependencies using pip:

```bash
pip install langchain-core langchain-openai nltk rich
```

You'll also need to download the `punkt` resource for `nltk`:
```python
import nltk
nltk.download('punkt')
```

## Usage

```bash
python crossword.py [options]
```

**Options:**

*   `--width`: Width of the grid (default: 15).
*   `--height`: Height of the grid (default: 15).
*   `--black_squares`: Approximate percentage of black squares (0.0 to 1.0, default: 0.17).
*   `--manual_grid`:  Manually specify the grid layout ('.'=white, '#'=black).  Example: `"...#...#......\n.....#........\n.............."`
*   `--grid_file`: Path to a file containing the grid layout (same format as `--manual_grid`).
*   `--lm_studio_url`:  LM Studio server URL (default: "http://localhost:1234/v1").
*   `--words_file`: Path to the word list file (default: "data/parole.txt").  Each line should contain a single word.
*   `--output_filename`: Output HTML filename (default: "docs/cruciverba.html").
*   `--max_attempts`: Maximum attempts to place a word (default: 100).
*   `--timeout`: Overall timeout for word selection (seconds, default: 180).
*   `--llm_timeout`: Timeout for LLM requests (seconds, default: 30).
*   `--llm_max_tokens`: Maximum tokens for LLM responses (default: 64).
*   `--language`: Language for definitions (default: "Italian").
*   `--model`: Model name to use for definitions (default: "mistralai/Mistral-7B-Instruct-v0.1").
*   `--max_grid_iterations`: Maximum number of attempts to generate a complete grid (default: 5).
*    `--difficulty`: Difficulty level, influencing word frequency preference ('easy', 'medium', 'hard', default: 'medium').

**Example:**

To generate a 20x20 crossword with a higher density of black squares (25%), using the English word list "words.txt", and saving the output to "my_crossword.html", you would run:

```bash
python crossword.py --width 20 --height 20 --black_squares 0.25 --words_file words.txt --output_filename my_crossword.html --language English
```

**Word List Format:**

The word list file should contain one word per line, with no additional punctuation or spaces.  Example:

```
apple
banana
cherry
...
```

**LM Studio Setup:**

This script is designed to work with [LM Studio](https://lmstudio.ai/), a tool for running local LLMs.  You'll need to have LM Studio installed and a suitable model loaded (Mistral-7B-Instruct-v0.1 is recommended).  Ensure that the LM Studio server is running at the URL specified by `--lm_studio_url` (the default is `http://localhost:1234/v1`).

**Manual Grid Input:**

The `--manual_grid` option allows you to define the grid structure directly.  Use `.` for white squares and `#` for black squares.  Each row should be on a new line, and all rows must have the same length.  The `--grid_file` option works similarly, but reads the grid layout from a file.

**Output:**

The script generates an HTML file (`cruciverba.html` by default) containing the interactive crossword puzzle.  Open this file in a web browser to play the crossword.  The generated HTML is self-contained and doesn't require an internet connection (after the initial generation).
The javascript code checks for the correct solution.

**Troubleshooting:**

*   **"No valid words found" error:** Ensure your word list file exists and is formatted correctly.
*   **"Failed to generate definition" errors:** The LLM may occasionally fail to generate a suitable clue.  The script will retry a few times, but some words might not have definitions.
*   **Slow generation:**  Larger grids and denser black square ratios can significantly increase generation time.  Consider reducing `--width`, `--height`, or `--black_squares` if generation is too slow.  You can also increase --timeout.
*  **"Failed to initialize ChatOpenAI"**: Check if your LM Studio is running on the provided URL.
* **"template.html not found"**: be sure to have template.html in the same crossword.py directory.
