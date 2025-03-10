# Crosswords

This Python script generates interactive crossword puzzles in HTML format. It leverages LangChain for definition generation and provides various options for grid creation, word selection, and output customization.

- ## [Try the demo](https://fabriziosalmi.github.io/crosswords/)

## Features

*   **Grid Generation:**
    *   Random grid generation with a specified black square ratio.
    *   Manual grid definition via a string.
    *   Loading grid layouts from a file.
*   **Word Selection:**
    *   Loads words from a provided text file (one word per line).
    *   Sophisticated word placement algorithm with constraint checking and backtracking to ensure valid crossword construction.
    *   Heuristic-based slot selection (prioritizes constrained slots).
    *   Timeout and maximum attempts to prevent infinite loops.
*   **Definition Generation:**
    *   Uses LangChain with a local LM Studio instance (or any compatible OpenAI API endpoint) to generate crossword-style definitions.
    *   Definition caching to improve performance.
    *   Definition cleaning and validation to avoid revealing the answer.
    *   Multi-threaded definition generation for speed.
*   **Interactive HTML Output:**
    *   Generates a fully interactive HTML crossword puzzle.
    *   Arrow key navigation between cells.
    *   Automatic capitalization of input.
    *   Clear visual distinction between black squares and input cells.
    *   Organized presentation of definitions (across and down).
*   **Error Handling:**
    *   Extensive input validation.
    *   Robust error handling and logging throughout the process.
    *   Early exit if grid is determined to be unsolvable.
    *    Handles scenarios where no valid word fits for a slot.
    *    Recursion depth limit to prevent stack overflows.
*    **Progress Bars:**
     * Uses rich library for progress visualization.

## Requirements

*   Python 3.7+
*   `langchain_core`
*   `langchain_openai`
*   `rich`
*   An accessible LM Studio instance (or a compatible OpenAI API endpoint).
*   A text file containing a list of words (one word per line) – a sample file (`data/parole.txt`) should be created in the same directory as the script.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/fabriziosalmi/crosswords
    cd crosswords
    ```
2.  Install the required packages:
    ```bash
    pip install langchain-core langchain-openai rich
    ```
## Usage

```bash
python crossword.py [options]
```

Replace `crossword.py` with the actual name of your Python script.

**Options:**

*   `--width`: Width of the grid (columns).  Default: 4
*   `--height`: Height of the grid (rows). Default: 3
*   `--black_squares`: Approximate percentage of black squares (0.0 to 1.0). Default: 0.2
*   `--manual_grid`: Manually specify the grid layout ('.'=white, '#'=black).  Example: `"..#..\n....."`
*   `--grid_file`: Path to a file containing the grid layout (same format as `manual_grid`).
*   `--lm_studio_url`: The URL of your LM Studio server. Default: `http://localhost:1234/v1`
*   `--words_file`: Path to the file containing the words. Default: `data/parole.txt`
*   `--output_filename`: Output HTML filename. Default: `docs/cruciverba.html`
*   `--max_attempts`: Maximum attempts to place a word in a slot. Default: 50
*   `--timeout`: Timeout for word selection (in seconds). Default: 60
*   `--llm_timeout`: Timeout for LLM requests (in seconds). Default: 30
*   `--llm_max_tokens`: Maximum number of tokens for LLM responses. Default: 48
*   `--language`: Language for the definitions. Default: `Italian`

**Example:**

To generate a 10x10 crossword with approximately 25% black squares, using the words from `mywords.txt` and saving the output to `puzzle.html`, you would run:

```bash
python crossword.py --width 10 --height 10 --black_squares 0.25 --words_file mywords.txt --output_filename puzzle.html
```

To use a pre-defined grid from a file:

```bash
python crossword.py --grid_file mygrid.txt --words_file mywords.txt
```

Where `mygrid.txt` contains something like:

```
..#..
.....
#....
.....
..#..
```

## Important Notes

*   **LM Studio Setup:** Ensure LM Studio (or your chosen OpenAI-compatible endpoint) is running and accessible at the specified URL.
*   **Words File:**  The words file should contain one word per line, and ideally be pre-processed (e.g., all uppercase, no punctuation).  The quality of the word list greatly impacts the success of crossword generation.
*   **Grid Design:**  Creating good crossword grids is an art.  The `--manual_grid` and `--grid_file` options give you control, but good grid design is non-trivial. The random generator is a reasonable starting point.  Very small grids or grids with very few/many black squares can be difficult or impossible to fill.
*   **Timeouts:**  The `--timeout` and `--llm_timeout` parameters are crucial for preventing the script from running indefinitely if it cannot find a solution.  Adjust these based on grid size and complexity.
* **Model:** The code is preconfigured to use "deephermes-3-llama-3-8b-preview", change accordingly with your model.

## Troubleshooting

*   **"No words of length X found" error:**  The script cannot find any words in your `words_file` that match the required length for a particular slot.  Ensure your word list has sufficient variety.
*   **"Failed to generate a complete crossword puzzle" message:**  The script could not find a valid combination of words to fill the entire grid within the given constraints (timeout, max attempts).  Try increasing the timeout, increasing max_attempts, using a different grid, or providing a larger word list.
*   **"Error generating definition" errors:**  This indicates a problem with the LLM interaction. Check your LM Studio setup, network connection, and the `lm_studio_url` parameter.
* **"Invalid grid dimensions: inconsistent row length" error:** Check that you respect the correct format inside your --grid_file or --manual_grid options.
* **"The generated grid does not have enough valid slots" error:** If using --grid_file or --manual_grid options, make sure the provided grid contains at least two valid word placements.
