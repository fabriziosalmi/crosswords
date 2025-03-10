import random
import argparse
from typing import List, Tuple, Dict
from rich.progress import Progress, TaskID
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import string
import time
import logging

# --- LangChain Imports (Corrected) ---
from langchain_core.prompts import PromptTemplate  # Use langchain_core
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
# from langchain.callbacks.manager import CallbackManager # Not needed for this example
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # Not needed


# Constants
DEFAULT_WIDTH = 8
DEFAULT_HEIGHT = 5
DEFAULT_BLACK_SQUARES = 0.18
LM_STUDIO_URL = "http://localhost:1234/v1"  # Correct URL (remove /completions)
DEFAULT_WORDS_FILE = "parole.txt"
DEFAULT_OUTPUT_FILENAME = "cruciverba.html"
DEFAULT_MAX_ATTEMPTS = 50
DEFAULT_TIMEOUT = 60
DEFAULT_LLM_TIMEOUT = 30
MAX_RETRIES = 3  # Retries are handled by LangChain, but we can set a default

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- LangChain Setup (Corrected) ---
def setup_langchain_llm(lm_studio_url: str, llm_timeout: int):
    """Sets up the LangChain LLM (ChatOpenAI) with retries and timeout."""

    llm = ChatOpenAI(
        base_url=lm_studio_url,  # Use the correct base URL
        api_key="NA",  # LM Studio doesn't require an API key
        model="deephermes-3-llama-3-8b-preview", # Specify a model name (can be anything)
        temperature=0.7,
        max_tokens=32,
        timeout=llm_timeout, # Use LangChain's timeout
    )
    return llm


# --- Definition Generation with LangChain (Corrected) ---

def generate_definition_langchain(llm: ChatOpenAI, word: str) -> str:
    """Generates a definition using LangChain."""

    prompt_template = """Genera una definizione breve e adatta a un cruciverba per la parola: {word}"""
    prompt = PromptTemplate.from_template(prompt_template)
    output_parser = StrOutputParser()
    # Use RunnablePassthrough to pass the input directly
    chain = (
        {"word": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
     )

    try:
        definition = chain.invoke(word) # Use invoke
        definition = definition.strip()
        definition = definition.replace(f"Definizione di {word}:", "").strip()
        definition = definition.replace(f"Definizione per '{word}':", "").strip()
        definition = definition.replace(f"Definizione:", "").strip()
        definition = re.sub(r'^\d+\.\s*', '', definition)  # Remove leading numbers
        if definition.startswith('"') and definition.endswith('"'):
            definition = definition[1:-1]
        return definition

    except Exception as e:
        logging.error(f"Error generating definition for {word}: {e}")
        return "Definizione non disponibile"



def _generate_definition(llm: ChatOpenAI, word: str, row: int, col: int, direction: str, numbering: dict,
                           next_number_ref: List[int]) -> Tuple[str, int, int, str, str]:
    """Helper function using LangChain (for threading)."""
    if (row, col, direction) not in numbering:
        numbering[(row, col, direction)] = next_number_ref[0]
        next_number_ref[0] += 1

    definition = generate_definition_langchain(llm, word)
    return direction, numbering[(row, col, direction)], word, definition


def generate_definitions(placed_words: List[Tuple[str, int, int, str]], llm: ChatOpenAI) -> Dict[str, Dict[int, str]]:
    """Generates definitions using threading and LangChain."""
    definitions = {"across": {}, "down": {}}
    numbering = {}
    next_number_ref = [1]

    with Progress() as progress:
        task = progress.add_task("[blue]Generating Definitions...", total=len(placed_words))
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(_generate_definition, llm, word, row, col, direction, numbering, next_number_ref)
                       for word, row, col, direction in placed_words]

            for future in as_completed(futures):
                try:
                    direction, num, word, definition = future.result()
                    definitions[direction][f"{num}. {word}"] = definition
                except Exception as e:
                    logging.error(f"Error retrieving definition: {e}")
                finally:
                    progress.update(task, advance=1)
    return definitions



# --- Keep your other functions (generate_grid, find_slots, etc.) as they are ---
# ... (rest of your functions: generate_grid, find_slots, load_words,
# ... is_valid_placement, place_word, check_all_letters_connected,
# ... slots_intersect, calculate_intersections, select_words_recursive,
# ... select_words, create_html) ...

def generate_grid(width: int, height: int, black_square_ratio: float) -> List[List[str]]:
    """Generates a random crossword grid."""
    grid = [['.' for _ in range(width)] for _ in range(height)]
    num_black_squares = int(width * height * black_square_ratio)

    with Progress() as progress:
        task1 = progress.add_task("[red]Generating Grid...", total=num_black_squares)
        for _ in range(num_black_squares // 2):
            while True:
                row = random.randint(0, height - 1)
                col = random.randint(0, width - 1)
                if grid[row][col] == '.' and grid[height - 1 - row][width - 1 - col] == '.':
                    grid[row][col] = '#'
                    grid[height - 1 - row][width - 1 - col] = '#'
                    progress.update(task1, advance=2)  # Advance by 2 for symmetry
                    break
        remaining = num_black_squares - 2 * (num_black_squares // 2)
        for _ in range(remaining):
            while True:
                row = random.randint(0, height - 1)
                col = random.randint(0, width - 1)
                if grid[row][col] == '.':
                    grid[row][col] = '#'
                    progress.update(task1, advance=1)
                    break
    return grid


def find_slots(grid: List[List[str]]) -> List[Tuple[int, int, str, int]]:
    """Finds slots for words in the grid."""
    height = len(grid)
    width = len(grid[0])
    slots = []
    with Progress() as progress:
        task = progress.add_task("[cyan]Finding Slots...", total=height + width)

        for row in range(height):
            start = -1
            for col in range(width):
                if grid[row][col] == '.':
                    if start == -1:
                        start = col
                elif start != -1:
                    length = col - start
                    if length > 1:
                        slots.append((row, start, "across", length))
                    start = -1
            if start != -1:
                length = width - start
                if length > 1:
                    slots.append((row, start, "across", length))
            progress.update(task, advance=1)

        for col in range(width):
            start = -1
            for row in range(height):
                if grid[row][col] == '.':
                    if start == -1:
                        start = row
                elif start != -1:
                    length = row - start
                    if length > 1:
                        slots.append((start, col, "down", length))
                    start = -1
            if start != -1:
                length = height - start
                if length > 1:
                    slots.append((start, col, "down", length))
            progress.update(task, advance=1)
    return slots


def load_words(filepath: str) -> Dict[int, List[str]]:
    """Loads words, preprocesses, and groups by length."""
    words_by_length = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            with Progress() as progress:
                f.seek(0, 2)  # Move to the end of the file
                file_size = f.tell()  # Get the file size
                f.seek(0)  # Go back to the beginning
                task = progress.add_task("[green]Loading and Preprocessing Words...", total=file_size)
                for line in f:
                    word = line.strip().upper()
                    # Remove non-alphabetic characters and keep only A-Z
                    word = re.sub(r"[^A-Z]", "", word)

                    if not word:
                        continue

                    length = len(word)
                    if length > 1:
                        if length not in words_by_length:
                            words_by_length[length] = []
                        words_by_length[length].append(word)
                    #Advance based on encoded line size.
                    progress.update(task, advance=len(line.encode('utf-8')))

    except FileNotFoundError:
        logging.error(f"Word file not found at {filepath}")
        exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading words: {e}")
        exit(1)

    return words_by_length


def is_valid_placement(grid: List[List[str]], word: str, row: int, col: int, direction: str) -> bool:
    """Checks if a word can be placed at the given position."""
    length = len(word)
    if direction == "across":
        if col + length > len(grid[0]):
            return False
        for i in range(length):
            if grid[row][col + i] != '.' and grid[row][col + i] != word[i]:
                return False
    else:  # direction == "down"
        if row + length > len(grid):
            return False
        for i in range(length):
            if grid[row + i][col] != '.' and grid[row + i][col] != word[i]:
                return False
    return True


def place_word(grid: List[List[str]], word: str, row: int, col: int, direction: str) -> List[List[str]]:
    """Places a word in the grid (creates a copy)."""
    new_grid = [row[:] for row in grid]  # Deep copy
    length = len(word)
    if direction == "across":
        for i in range(length):
            new_grid[row][col + i] = word[i]
    else:
        for i in range(length):
            new_grid[row + i][col] = word[i]
    return new_grid


def check_all_letters_connected(grid: List[List[str]], placed_words: List[Tuple[str, int, int, str]]) -> bool:
    """Ensures all placed letters are part of both an across and a down word."""
    height = len(grid)
    width = len(grid[0])
    letter_grid = [['' for _ in range(width)] for _ in range(height)]
    placed_coords = set()

    # Build a grid of placed letters and track their coordinates.
    for word, row, col, direction in placed_words:
        for i, letter in enumerate(word):
            if direction == 'across':
                letter_grid[row][col + i] = letter
                placed_coords.add((row, col + i))  # Add coordinates
            else:
                letter_grid[row + i][col] = letter
                placed_coords.add((row + i, col)) # Add coordinates


    # Check each placed letter.
    for row, col in placed_coords:  # Iterate through placed coordinates
        in_across = False
        in_down = False
        for word, word_row, word_col, word_dir in placed_words:
            if word_dir == 'across' and word_row == row and word_col <= col < word_col + len(word):
                in_across = True
            elif word_dir == 'down' and word_col == col and word_row <= row < word_row + len(word):
                in_down = True
        if not (in_across and in_down):
            return False  # Found a letter not in both across and down

    return True  # All letters are connected


def slots_intersect(slot1: Tuple[int, int, str, int], slot2: Tuple[int, int, str, int]) -> bool:
    """Checks if two slots intersect."""
    row1, col1, dir1, len1 = slot1
    row2, col2, dir2, len2 = slot2

    if dir1 == dir2:
        return False  # Parallel slots don't intersect

    if dir1 == "across":
        if col1 <= col2 < col1 + len1 and row2 <= row1 < row2 + len2:
            return True
    else:  # dir1 == "down"
        if row1 <= row2 < row1 + len1 and col2 <= col1 < col2 + len2:
            return True

    return False


def calculate_intersections(slots: List[Tuple[int, int, str, int]]) -> Dict[int, List[int]]:
    """Calculates which slots intersect with each other."""
    intersections = {i: [] for i in range(len(slots))}
    for i in range(len(slots)):
        for j in range(i + 1, len(slots)):
            if slots_intersect(slots[i], slots[j]):
                intersections[i].append(j)
                intersections[j].append(i)
    return intersections


def select_words_recursive(grid: List[List[str]], slots: List[Tuple[int, int, str, int]],
                           words_by_length: Dict[int, List[str]],
                           placed_words: List[Tuple[str, int, int, str]],
                           max_attempts: int, start_time: float,
                           timeout: int, progress: Progress, task: TaskID) -> Tuple[List[List[str]], List[Tuple[str, int, int, str]], int]:
    """Recursive word selection with progress updates and optimizations."""

    if time.time() - start_time > timeout:
        return None, None, 0

    if not slots:
        if check_all_letters_connected(grid, placed_words):
            return grid, placed_words, 0
        else:
            return None, None, 0

    best_slot = None
    min_valid_words = float('inf')

    for slot in slots:
        row, col, direction, length = slot
        possible_words = words_by_length.get(length, [])
        valid_word_count = 0
        for word in possible_words:
            if is_valid_placement(grid, word, row, col, direction):
                valid_word_count += 1
        if valid_word_count < min_valid_words:
            min_valid_words = valid_word_count
            best_slot = slot

    current_slot = best_slot
    row, col, direction, length = current_slot
    remaining_slots = [s for s in slots if s != current_slot]
    possible_words = words_by_length.get(length, [])

    allowed_letters = [set(string.ascii_uppercase) for _ in range(length)]
    for word, placed_row, placed_col, placed_dir in placed_words:
        if direction == "across" and placed_dir == "down":
            if placed_row <= row < placed_row + len(word) and col <= placed_col < col + length:
                intersect_index = placed_col - col
                allowed_letters[intersect_index] = {word[row - placed_row]}
        elif direction == "down" and placed_dir == "across":
            if placed_col <= col < placed_col + len(word) and row <= placed_row < row + length:
                intersect_index = placed_row - row
                allowed_letters[intersect_index] = {word[col - placed_col]}

    filtered_possible_words = [
        word for word in possible_words
        if all(letter in allowed_letters[i] for i, letter in enumerate(word))
    ]
    random.shuffle(filtered_possible_words)
    dynamic_max_attempts = min(max_attempts * 2, max(10, len(filtered_possible_words) * 2))
    attempts_count = 0

    for word in filtered_possible_words:
        if is_valid_placement(grid, word, row, col, direction):
            attempts_count += 1
            progress.update(task, advance=1)  # Update progress *per attempt*
            new_grid = place_word(grid, word, row, col, direction)
            new_placed_words = placed_words + [(word, row, col, direction)]

            lookahead_valid = True
            for temp_slot in remaining_slots:
                temp_row, temp_col, temp_dir, temp_length = temp_slot
                temp_possible_words = words_by_length.get(temp_length, [])
                if not any(is_valid_placement(new_grid, temp_word, temp_row, temp_col, temp_dir) for temp_word in temp_possible_words):
                    lookahead_valid = False
                    break

            if lookahead_valid:
                result_grid, result_placed_words, recursive_attempts = select_words_recursive(
                    new_grid, remaining_slots, words_by_length, new_placed_words,
                    dynamic_max_attempts, start_time, timeout, progress, task
                )
                attempts_count += recursive_attempts
                if result_grid:
                    return result_grid, result_placed_words, attempts_count

    return None, None, attempts_count


def select_words(grid: List[List[str]], slots: List[Tuple[int, int, str, int]],
                 words_by_length: Dict[int, List[str]], progress: Progress, task: TaskID,
                 timeout: int = DEFAULT_TIMEOUT) -> Tuple[
    List[List[str]], List[Tuple[str, int, int, str]]]:
    """Wrapper for recursive function, sorts slots, and manages progress."""

    intersections = calculate_intersections(slots)
    sorted_slots = sorted(slots, key=lambda x: (len(intersections.get(slots.index(x), [])), x[3]), reverse=True)
    start_time = time.time()

    # Estimate total attempts (this will likely be an underestimate)
    total_estimated_attempts = sum(len(words_by_length.get(slot[3], [])) for slot in sorted_slots)
    progress.update(task, total=total_estimated_attempts)

    filled_grid, placed_words, total_attempts = select_words_recursive(
        grid, sorted_slots, words_by_length, [], DEFAULT_MAX_ATTEMPTS, start_time, timeout, progress, task
    )

    return filled_grid, placed_words

def create_html(grid: List[List[str]], placed_words: List[Tuple[str, int, int, str]], definitions: Dict[str, Dict[int, str]], filename: str):
    """Generates the HTML file for the crossword puzzle."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write("<!DOCTYPE html>\n")
        f.write("<html lang='it'>\n")
        f.write("<head>\n")
        f.write("  <meta charset='UTF-8'>\n")
        f.write("  <title>Cruciverba</title>\n")
        f.write("  <style>\n")
        f.write("    table { border-collapse: collapse; margin-right: 20px; float: left; }\n")  # Float the table
        f.write("    td { border: 1px solid black; width: 30px; height: 30px; text-align: center; font-size: 20px; font-family: sans-serif; }\n")
        f.write("    .black { background-color: black; }\n")
        f.write("    .input-cell { font-weight: bold; }\n")
        f.write("    input { border: none; width: 28px; height: 28px; text-align: center; font-size: 20px; font-family: sans-serif; padding: 0; outline: none; }\n")  # Remove outline
        f.write("    input:focus { background-color: #e0e0e0; }\n")  # Optional: Highlight on focus
        f.write("  </style>\n")
        f.write("</head>\n")
        f.write("<body>\n")
        f.write("  <h1>Cruciverba</h1>\n")
        f.write("  <div style='display: flex;'>\n")  # Use flexbox for side-by-side layout
        f.write("    <table>\n")

        # Create a numbering map
        numbering = {}
        next_number = 1
        for word, row, col, direction in placed_words:
            if (row, col, direction) not in numbering:
                numbering[(row, col, direction)] = next_number
                next_number += 1

        # Write table
        for r, row in enumerate(grid):
            f.write("  <tr>\n")
            for c, cell in enumerate(row):
                if cell == '#':
                    f.write("    <td class='black'></td>\n")
                else:
                    # Check if this cell is the start of a word
                    cell_number = ""
                    for (word_row, word_col, word_dir), num in numbering.items():
                        if word_row == r and word_col == c:
                            cell_number = str(num)
                            break

                    f.write(f"    <td>{cell_number}<input type='text' maxlength='1' oninput='this.value=this.value.toUpperCase()'></td>\n")

            f.write("  </tr>\n")
        f.write("    </table>\n")


        # Definitions
        f.write("   <div style='float: left;'>\n")  # Definitions container
        f.write("     <h2>Definizioni</h2>\n")

        f.write("     <h3>Orizzontali</h3>\n")
        f.write("     <ul>\n")
        for key, definition in definitions["across"].items():
              f.write(f"       <li><b>{key}</b> {definition}</li>\n")
        f.write("     </ul>\n")

        f.write("     <h3>Verticali</h3>\n")
        f.write("     <ul>\n")
        for key, definition in definitions["down"].items():
              f.write(f"       <li><b>{key}</b> {definition}</li>\n")
        f.write("     </ul>\n")
        f.write("   </div>\n")  # Close definitions div

        f.write("  </div>\n")  # Close flex container
        f.write("</body>\n")
        f.write("</html>\n")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generates an interactive crossword puzzle.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Grid width.")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Grid height.")
    parser.add_argument("--black_squares", type=float, default=DEFAULT_BLACK_SQUARES, help="Percentage of black squares.")
    parser.add_argument("--lm_studio_url", type=str, default=LM_STUDIO_URL, help="LM Studio URL.")
    parser.add_argument("--words_file", type=str, default=DEFAULT_WORDS_FILE, help="Path to the words file.")
    parser.add_argument("--output_filename", type=str, default=DEFAULT_OUTPUT_FILENAME, help="Output HTML filename.")
    parser.add_argument("--max_attempts", type=int, default=DEFAULT_MAX_ATTEMPTS, help="Maximum attempts per slot.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout for word selection in seconds.")
    parser.add_argument("--llm_timeout", type=int, default=DEFAULT_LLM_TIMEOUT, help="Timeout for LLM requests in seconds.")

    args = parser.parse_args()

    # --- LangChain LLM Setup (using the function) ---
    llm = setup_langchain_llm(args.lm_studio_url, args.llm_timeout)


    words_by_length = load_words(args.words_file)
    if not words_by_length:
        logging.error("No valid words found in the word file.")
        exit(1)

    grid = generate_grid(args.width, args.height, args.black_squares)
    slots = find_slots(grid)

    with Progress() as progress:
        task_select_words = progress.add_task("[yellow]Selecting Words...")
        filled_grid, placed_words = select_words(grid, slots, words_by_length, progress, task_select_words,
                                                  args.timeout)
        progress.stop()

    if filled_grid is not None:
        # --- Generate Definitions using LangChain ---
        definitions = generate_definitions(placed_words, llm)
        create_html(filled_grid, placed_words, definitions, args.output_filename)
        print(f"Crossword puzzle created and saved to {args.output_filename}")
    else:
        print("Failed to generate a complete crossword puzzle.")

if __name__ == "__main__":
    main()