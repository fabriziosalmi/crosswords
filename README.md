# Interactive Crossword Puzzle Generator

This Python script (`crossword.py`) generates interactive crossword puzzles in HTML format, leveraging Language Models (LLMs) for clue generation.  

## Introduction

`crossword.py` creates crossword puzzles from scratch, handling everything from grid generation and word placement to clue generation and the creation of an interactive HTML interface. It offers extensive customization options, allowing you to tailor the puzzle to your specific needs.  The core algorithm combines constraint satisfaction techniques with a sophisticated backtracking search, enhanced by features like beam search, dynamic difficulty adjustment, and optional multi-threading.

## Features

*   **Highly Customizable Grid Generation:**
    *   **Dimensions:** Specify the width and height of the grid using `--width` and `--height`.
    *   **Black Square Ratio:** Control the density of black squares with `--black_squares` (a float between 0.0 and 1.0). The script also *dynamically adjusts* this ratio during generation to improve solvability.
    *   **Manual Grid Input:** Define the grid layout precisely using `--manual_grid` (providing a string representation) or `--grid_file` (pointing to a text file).  This is ideal for creating specific puzzle shapes or incorporating pre-existing grid designs.
    *   **Symmetry:** Randomly generated grids are automatically symmetrical (180-degree rotational symmetry), a standard convention in crossword design.
    * **Grid Validation**: Ensures the manual grids provided are correct

*   **Intelligent Word Placement:**
    *   **Word List:**  Provide a plain text file containing your word list (one word per line) using `--words_file`. The script preprocesses this list, calculates word frequencies, and builds a highly optimized index for fast lookup.
    *   **Minimum Word Length:**  Filters out words shorter than a minimum length (default: 2, to avoid trivial solutions).
    *   **Difficulty Levels:** Select a difficulty level (`--difficulty easy`, `--difficulty medium`, or `--hard`) to influence word choice.  "Easy" prioritizes common words, "hard" prefers rarer words, and "medium" strikes a balance.
    *   **Constraint Satisfaction:** The algorithm uses constraint propagation (forward checking) to ensure that every word placement leaves viable options for intersecting words. This drastically reduces dead ends during the search.
    *   **Backtracking Search:** A sophisticated backtracking search explores the solution space.  It includes:
        *   **Beam Search:** Limits the number of candidate words considered at each step (`--beam_width`, dynamically adjusted).
        *   **Dynamic Backtracking Limit:**  Adjusts the maximum number of backtracks (`--max_backtrack`, dynamically adjusted) during the search to balance exploration and efficiency.
        *   **Slot Ordering:**  Prioritizes filling the most constrained slots first (those with the fewest possible word choices), significantly improving performance.
        * **Intersection Score**: calculate a score for intersections.
        * **Location Score**: calculates a bonus for location and shape of the placed word.
    *   **Parallel Processing (Top-Level):** Employs a `ThreadPoolExecutor` to explore multiple word placement options concurrently at the top level of the search tree, substantially accelerating the generation process (controlled by `MAX_THREAD_POOL_SIZE`).  Recursive calls within the search use a single thread to avoid excessive overhead.
    * **All Letters Connected**: Checks that all placed letters are connected in across and down words.
    * **Avoids 2x2 white spaces blocks:** by checking the grid before placing black squares.

*   **LLM-Powered Clue Generation:**
    *   **LM Studio Integration:** Seamlessly integrates with [LM Studio](https://lmstudio.ai/) for local LLM inference.  Specify the server URL using `--lm_studio_url` (defaults to `http://localhost:1234/v1`).
    *   **Model Choice:** Select the LLM model you want to use via `--model` (defaults to `meta-llama-3.1-8b-instruct`).
    *   **Customizable Prompt:**  Uses a carefully crafted prompt (within `generate_definition_langchain`) to guide the LLM towards generating concise, crossword-style clues.  This prompt enforces strict rules to avoid trivial or invalid clues.
    *   **Robust Filtering:**  Filters out clues that contain the answer word, parts of the answer word, or obvious synonyms.  This ensures clue quality and prevents the LLM from "cheating."
    *   **Retries:**  Automatically retries clue generation multiple times (`MAX_DEFINITION_ATTEMPTS`) with a short delay (`DEFINITION_RETRY_DELAY`) if the LLM fails to produce a valid clue.
    *   **Caching:** Caches generated definitions to avoid redundant LLM calls, significantly improving performance, especially for larger puzzles or repeated runs.
    *   **Langchain Integration:** leverages Langchain.

*   **Interactive HTML Output:**
    *   **Self-Contained:** Generates a single, self-contained HTML file that can be opened directly in any modern web browser. No external dependencies or internet connection are required to play the puzzle.
    *   **User-Friendly Interface:**  Provides a clean, intuitive interface for entering answers.
    *   **Clue Numbering:**  Automatically numbers clues according to standard crossword conventions (top-to-bottom, left-to-right).
    *   **Solution Checking:**  Includes JavaScript code to validate user input against the correct solution (client-side validation).

*   **Comprehensive Logging and Statistics:**
    *   **Detailed Output:** Uses the `rich` library to provide visually informative console output, including progress bars, grid previews, and detailed statistics.
    *   **Performance Metrics:** Tracks and reports key metrics like the number of attempts, backtracks, successful/failed placements, definition failures, cache hits/misses, and overall generation time.  This data is invaluable for understanding the algorithm's behavior and optimizing parameters.
    *   **Dynamic Parameter Tracking:**  Displays the current values of dynamically adjusted parameters (beam width and maximum backtrack limit) during the generation process.
    *   **Slot Fill Order Logging:**  Records the order in which slots were filled, providing insight into the search path.

*   **Error Handling and Input Validation:**
    *   **Robust Input Validation:**  Thoroughly validates all command-line arguments to prevent common errors and provide informative error messages.
    *   **Graceful Failure:** Handles potential errors (e.g., file not found, LLM connection issues) gracefully, providing informative error messages and exiting cleanly.
    * **Grid File and Manual Grid Handling**: Reads grids from files, parses strings, and handles invalid formats with informative warnings.

## Requirements

*   Python 3.7+
*   `langchain-core`
*   `langchain-openai`
*   `rich`

Install the dependencies using pip:

```bash
pip install langchain-core langchain-openai rich
```

## Usage

```bash
python crossword.py [options]
```

**Options:**

| Option                | Description                                                                                                                                                                                            | Default                      |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------- |
| `--width`             | Width of the grid (columns).                                                                                                                                                                         | `4`                           |
| `--height`            | Height of the grid (rows).                                                                                                                                                                          | `4`                           |
| `--black_squares`     | Approximate percentage of black squares (0.0 to 1.0).                                                                                                                                                | `0.2`                        |
| `--manual_grid`       | Manually specify the grid layout ('.'=white, '#'=black).  Example: `"...#...#......\n.....#........\n.............."`                                                                                  | `None`                       |
| `--grid_file`         | Path to a file containing the grid layout (same format as `--manual_grid`).                                                                                                                         | `None`                       |
| `--lm_studio_url`     | LM Studio server URL.                                                                                                                                                                                 | `"http://localhost:1234/v1"` |
| `--words_file`        | Path to the word list file (one word per line).                                                                                                                                                      | `"data/parole.txt"`          |
| `--output_filename`   | Output HTML filename.                                                                                                                                                                                  | `"docs/cruciverba.html"`     |
| `--max_attempts`      | Maximum attempts to place a word (per word placement, *not* global).                                                                                                                              | `10000`                       |
| `--timeout`           | Overall timeout for word selection (seconds).                                                                                                                                                        | `180`                        |
| `--llm_timeout`       | Timeout for LLM requests (seconds).                                                                                                                                                                   | `30`                         |
| `--llm_max_tokens`    | Maximum tokens for LLM responses.                                                                                                                                                                     | `64`                         |
| `--language`          | Language for definitions.                                                                                                                                                                              | `"Italian"`                  |
| `--model`             | Model name to use for definitions.                                                                                                                                                                     | `"meta-llama-3.1-8b-instruct"`   |
| `--max_grid_iterations` | Maximum number of attempts to generate a complete grid.                                                                                                                                             | `5`                          |
| `--difficulty`        | Difficulty level: 'easy', 'medium', or 'hard'.  Influences word frequency preference.                                                                                                                    | `"medium"`                   |

**Example:**

To generate a 15x15 crossword with a higher density of black squares (25%), using an English word list "words.txt", limiting LLM response tokens to 128, and saving the output to "my_crossword.html":

```bash
python crossword.py --width 15 --height 15 --black_squares 0.25 --words_file words.txt --output_filename my_crossword.html --language English --llm_max_tokens 128
```

**Word List Format:**

The word list file (`--words_file`) should be a plain text file with one word per line, using uppercase letters and no punctuation. Example:

```
APPLE
BANANA
CHERRY
...
```

**LM Studio Setup:**

1.  **Install LM Studio:** Download and install LM Studio from [https://lmstudio.ai/](https://lmstudio.ai/).
2.  **Load a Model:**  Open LM Studio and download/load a suitable language model.  The default model in `crossword.py` (`meta-llama-3.1-8b-instruct`) is a good starting point, but you can experiment with others.
3.  **Start the Server:**  In LM Studio, navigate to the "Local Inference Server" tab (usually the icon on the left that looks like `<->`). Select your loaded model, and click "Start Server."  Make sure the server is running on the address specified by `--lm_studio_url` (the default is `http://localhost:1234/v1`).

**Manual Grid Input:**

*   **`--manual_grid`:**  Provide the grid layout as a string, with `.` representing white squares and `#` representing black squares.  Each row should be separated by a newline character (`\n`).  For example:

    ```bash
    python crossword.py --manual_grid "....#....\n....#....\n....#....\n#########\n....#....\n....#....\n....#...."
    ```

*   **`--grid_file`:**  Create a text file (e.g., `my_grid.txt`) containing the grid layout in the same format as `--manual_grid`. Then, use the `--grid_file` option:

    ```bash
    python crossword.py --grid_file my_grid.txt
    ```

**Troubleshooting:**

*   **"No valid words found" error:**
    *   Ensure your word list file (`--words_file`) exists at the specified path.
    *   Check that the word list file is formatted correctly (one word per line, uppercase, no punctuation).
    *   Try a different word list or adjust the `--difficulty` setting.  A more restrictive difficulty (e.g., "hard") with a small word list might result in no valid words.

*   **"Failed to generate definition" errors:**
    *   The LLM may occasionally struggle to generate a clue for a particular word. The script retries several times, but some words might not get definitions.
    *   Try a different LLM model (`--model`).  Some models are better at clue generation than others.
    *   Ensure LM Studio is running and the specified model is loaded and serving requests.
    *   Check your network connection (if using a remote LLM).
    *   Reduce `--llm_max_tokens` if the LLM is producing overly long or truncated responses.  Increase it if definitions are consistently cut off.

*   **Slow generation:**
    *   Larger grids (`--width`, `--height`) and higher black square densities (`--black_squares`) significantly increase generation time.  The complexity grows exponentially with grid size.
    *   Reduce the grid size or black square ratio.
    *   Increase the overall `--timeout`.
    *   Experiment with different `--difficulty` settings.  "Easy" mode, which favors common words, can sometimes be faster.
    *  If you modify the code, make sure `MAX_THREAD_POOL_SIZE` is set appropriately for your system.  Too many threads can lead to performance degradation.

*   **"Failed to initialize ChatOpenAI" error:**
    *   Verify that LM Studio is running and that the local server is active.
    *   Double-check the `--lm_studio_url` to ensure it matches the address where LM Studio is serving the model.
    *   Ensure that the selected model (`--model`) is loaded in LM Studio.

*   **"template.html not found" error:**
    *   Make sure the `template.html` file is in the same directory as the `crossword.py` script.  This file is required for generating the interactive HTML output.

*   **Grid generation fails repeatedly (many "Failed to fill the grid completely" messages):**
    *   The combination of grid size, black square ratio, and word list might be too restrictive, making it impossible to find a valid solution.  Try:
        *   Reducing `--black_squares`.
        *   Increasing `--width` or `--height`.
        *   Using a larger and more diverse word list.
        *   Switching to `--difficulty easy`.
    *   Increase `--max_grid_iterations` to allow the script to try generating more grids before giving up.

*   **"Inconsistent row length in manual grid" or "Invalid manual grid" errors:**
    *   When using `--manual_grid` or `--grid_file`, ensure that all rows have the *exact same length*.
    *   The grid should only contain `.` (for white squares) and `#` (for black squares).  No other characters are allowed.
    *   Make sure the grid is rectangular (all rows have the same length, and all columns have the same height).

* **Crossword is generated, but some clues are missing:**
    * This is usually due to the LLM failing to generate a valid clue after multiple retries. It's a limitation of the LLM, and some words are inherently difficult to create good clues for.

## Output

The script generates a single HTML file (named `docs/cruciverba.html` by default, or specified by `--output_filename`). This file contains:

*   **Interactive Crossword Grid:** A visual representation of the crossword grid, with input fields for users to enter their answers.
*   **Numbered Clues:**  Lists of clues for "Across" and "Down" words, numbered according to standard crossword conventions.
*   **JavaScript Validation:**  Embedded JavaScript code that automatically checks the user's answers against the correct solution as they type.  This provides immediate feedback without requiring a server.

The generated HTML file is self-contained and can be opened directly in any modern web browser.  It does not require an internet connection to function *after* the initial generation process (which uses LM Studio/LLM).

## Contributing

Contributions are welcome!  Here are some ways you can contribute:

*   **Report Bugs:**  If you encounter any issues, please open an issue on the GitHub repository, providing a detailed description of the problem, steps to reproduce it, and your system configuration (OS, Python version, etc.).
*   **Suggest Features:**  If you have ideas for new features or improvements, please open an issue and describe your suggestion.
*   **Submit Pull Requests:**  If you've implemented a bug fix or new feature, feel free to submit a pull request.  Please ensure your code follows the existing style and includes appropriate comments and documentation.
* **Improve Word Lists:** contribute by enhancing or creating word lists for different languages.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. (You'll need to create a LICENSE file and put the MIT license text in it.)
```

Key improvements and explanations in this extended version:

*   **More Detailed Troubleshooting:**  Expanded the troubleshooting section to cover more potential issues and provide more specific solutions.  This includes addressing problems with grid generation failures, LLM issues, and manual grid input errors.
*   **Clearer Output Description:**  Provides a more precise description of the contents of the generated HTML file.
*   **Contribution Guidelines:** Added a "Contributing" section, encouraging contributions and outlining how users can report bugs, suggest features, and submit pull requests.  This is crucial for open-source projects.
*   **License Information:**  Includes a "License" section, specifying the MIT License (which is a very common and permissive open-source license).  **You'll need to create a file named `LICENSE` in your project's root directory and paste the text of the MIT license into it.**  You can find the MIT license text easily online (e.g., on the [Open Source Initiative website](https://opensource.org/licenses/MIT)).

