import argparse
import logging
import random
import re
import string
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional
from rich.progress import Progress, TaskID

# --- LangChain Imports ---
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# --- Local Imports ---
from html_generator import create_html

# --- Constants ---
DEFAULT_GRID_WIDTH = 3
DEFAULT_GRID_HEIGHT = 3
DEFAULT_BLACK_SQUARE_RATIO = 0.3
DEFAULT_LM_STUDIO_URL = "http://localhost:1234/v1"
DEFAULT_WORDS_FILE = "data/parole.txt"
DEFAULT_OUTPUT_FILENAME = "docs/cruciverba.html"
DEFAULT_MAX_ATTEMPTS = 50
DEFAULT_TIMEOUT = 120
DEFAULT_LLM_TIMEOUT = 30
DEFAULT_LLM_MAX_TOKENS = 48
DEFAULT_LANGUAGE = "Italian"
DEFAULT_MODEL = "meta-llama-3.1-8b-instruct"
DEFAULT_MAX_GRID_ITERATIONS = 5
MAX_RECURSION_DEPTH = 100  # Prevent stack overflow
DEFAULT_BEAM_WIDTH = 5
DEFAULT_MAX_BACKTRACK = 200
MIN_WORD_LENGTH = 2
FORBIDDEN_PATTERNS = [
    r'\b{}\b',  # Exact word match
    r'{}'.format,  # Direct word usage
    r'(?i){}'.format,  # Case insensitive match
]

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Compile regex patterns
WORD_CLEAN_RE = re.compile(r"[^A-Z]")
DEFINITION_CLEAN_RE = re.compile(r"^\d+\.\s*")
NON_ALPHANUMERIC_RE = re.compile(r"^[^\w]+|[^\w]+$")


def setup_langchain_llm(
    lm_studio_url: str, 
    llm_timeout: int, 
    llm_max_tokens: int,
    model: str
) -> ChatOpenAI:
    """
    Sets up the LangChain LLM (ChatOpenAI) with retries and timeout.

    Args:
        lm_studio_url: The URL of the LM Studio instance.
        llm_timeout: Timeout for LLM requests in seconds.
        llm_max_tokens: Maximum number of tokens for the LLM response.
        model: Model name to use.

    Returns:
        A ChatOpenAI instance.
    """
    try:
        llm = ChatOpenAI(
            base_url=lm_studio_url,
            api_key="NA",  # API key is not needed for local models
            model=model,
            temperature=0.7,
            max_tokens=llm_max_tokens,
            timeout=llm_timeout,
        )
        return llm
    except Exception as e:
        logging.error(f"Failed to initialize ChatOpenAI: {e}")
        sys.exit(1)


def generate_definition_langchain(
    llm: ChatOpenAI, word: str, language: str, definition_cache: Dict[str, str]
) -> str:
    """Enhanced definition generation with better filtering."""
    if word in definition_cache:
        return definition_cache[word]

    prompt_template = """Generate a short crossword clue for: {word}. Reply in {language}.
    IMPORTANT RULES:
    1. NEVER use the word itself or any part of it in the clue
    2. Keep it under 10 words
    3. No synonyms that are too obvious
    4. No direct definitions
    Format: Just the clue, nothing else."""

    prompt = PromptTemplate.from_template(prompt_template)
    output_parser = StrOutputParser()
    chain = (
        {"word": RunnablePassthrough(), "language": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
    )

    try:
        definition = chain.invoke({"word": word, "language": language})
        definition = definition.strip()
        
        # Enhanced cleaning
        definition = re.sub(r'(?i)definizione[:\s]*', '', definition)
        definition = re.sub(r'(?i)clue[:\s]*', '', definition)
        definition = re.sub(r'^\d+[\.\)]\s*', '', definition)
        definition = definition.strip()

        # Check for word or parts in definition
        word_lower = word.lower()
        definition_lower = definition.lower()
        
        # Check various forbidden patterns
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern(word), definition, re.IGNORECASE):
                return "Definizione alternativa richiesta"

        # Check for word parts (3+ letters)
        for i in range(len(word) - 2):
            for j in range(i + 3, len(word) + 1):
                if word[i:j].lower() in definition_lower:
                    return "Definizione alternativa richiesta"

        definition_cache[word] = definition
        return definition

    except Exception as e:
        logging.error(f"Error generating definition for {word}: {e}")
        return "Definizione non disponibile"


def _generate_definition(
    llm: ChatOpenAI,
    word: str,
    row: int,
    col: int,
    direction: str,
    cell_numbers: Dict[Tuple[int, int, str], int],
    next_number_ref: List[int],
    language: str,
    definition_cache: Dict[str, str],
) -> Tuple[str, int, str, str]:
    """Helper to generate definitions (for threading)."""
    if (row, col, direction) not in cell_numbers:
        cell_numbers[(row, col, direction)] = next_number_ref[0]
        next_number_ref[0] += 1

    definition = generate_definition_langchain(llm, word, language, definition_cache)
    return direction, cell_numbers[(row, col, direction)], word, definition


def order_cell_numbers(slots: List[Tuple[int, int, str, int]]) -> Dict[Tuple[int, int, str], int]:
    """
    Orders cell numbers according to standard crossword rules:
    1. Top to bottom
    2. Left to right
    3. Number only cells that start a word
    """
    numbered_cells = set()
    cell_numbers = {}
    
    # First, collect all starting positions
    start_positions = []
    for row, col, direction, length in slots:
        if length >= MIN_WORD_LENGTH:  # Only consider valid word lengths
            start_positions.append((row, col))
            numbered_cells.add((row, col))
    
    # Sort positions top-to-bottom, left-to-right
    start_positions.sort()
    
    # Assign numbers
    current_number = 1
    for row, col in start_positions:
        # Find all slots starting at this position
        for slot_row, slot_col, direction, length in slots:
            if slot_row == row and slot_col == col and length >= MIN_WORD_LENGTH:
                cell_numbers[(row, col, direction)] = current_number
        current_number += 1

    return cell_numbers


def generate_definitions(
    placed_words: List[Tuple[str, int, int, str]], llm: ChatOpenAI, language: str
) -> Dict[str, Dict[int, str]]:
    """Enhanced definition generation with proper numbering."""
    definitions = {"across": {}, "down": {}}
    definition_cache = {}
    
    # First, order all cell numbers
    slots = [(word, row, col, direction) for word, row, col, direction in placed_words]
    cell_numbers = order_cell_numbers(slots)
    
    with Progress() as progress:
        task = progress.add_task("[blue]Generating Definitions...", total=len(placed_words))
        with ThreadPoolExecutor() as executor:
            futures = []
            for word, row, col, direction in placed_words:
                if len(word) >= MIN_WORD_LENGTH:  # Only process valid words
                    futures.append(
                        executor.submit(
                            generate_definition_langchain,
                            llm,
                            word,
                            language,
                            definition_cache,
                        )
                    )

            for future, (word, row, col, direction) in zip(as_completed(futures), placed_words):
                try:
                    definition = future.result()
                    number = cell_numbers.get((row, col, direction))
                    if number:  # Only add if we have a valid number
                        definitions[direction][f"{number}. {word}"] = definition
                except Exception as e:
                    logging.error(f"Error retrieving definition: {e}")
                progress.update(task, advance=1)
    
    return definitions


def generate_grid_from_string(grid_string: str) -> Optional[List[List[str]]]:
    """Generates a grid from a string ('..#..\n.....')."""
    lines = grid_string.strip().split("\n")
    grid: List[List[str]] = []
    for line in lines:
        row = [char for char in line.strip() if char in (".", "#")]
        if len(row) != len(lines[0]): # Check consistent row length
            logging.error("Invalid grid dimensions: inconsistent row length.")
            return None
        grid.append(row)

    if not grid:
        logging.error("Invalid grid: empty grid.")
    return grid


def generate_grid_from_file(filepath: str) -> Optional[List[List[str]]]:
    """Loads grid from file. '#' = black, '.' = white."""
    try:
        with open(filepath, "r") as f:
            grid_string = f.read()
            return generate_grid_from_string(grid_string)
    except FileNotFoundError:
        logging.error(f"Grid file not found: {filepath}")
        return None
    except Exception as e:
        logging.error(f"Error reading grid from file: {e}")
        return None


def generate_grid_random(
    width: int, height: int, black_square_ratio: float
) -> List[List[str]]:
    """Generates a random crossword grid with improved patterns."""
    grid = [["." for _ in range(width)] for _ in range(height)]
    num_black_squares = int(width * height * black_square_ratio)
    
    # Ensure symmetrical pattern
    def add_symmetric_squares(row: int, col: int):
        grid[row][col] = "#"
        # Add symmetrical square
        grid[height - 1 - row][width - 1 - col] = "#"
        
    with Progress() as progress:
        task1 = progress.add_task("[red]Generating Grid...", total=num_black_squares//2)
        placed = 0
        max_attempts = 1000
        attempts = 0
        
        while placed < num_black_squares//2 and attempts < max_attempts:
            row = random.randint(0, height - 1)
            col = random.randint(0, width - 1)
            
            # Check if placement creates valid patterns
            if grid[row][col] == "." and grid[height-1-row][width-1-col] == ".":
                # Check surrounding cells
                valid = True
                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < height and 0 <= new_col < width:
                        if grid[new_row][new_col] == "#":
                            # Don't create 2x2 black square blocks
                            valid = False
                            break
                
                if valid:
                    add_symmetric_squares(row, col)
                    placed += 1
                    progress.update(task1, advance=1)
            
            attempts += 1
            
        # If we couldn't place enough black squares, try again
        if placed < num_black_squares//2:
            return generate_grid_random(width, height, black_square_ratio)
            
    return grid


def generate_grid(
    width: int,
    height: int,
    black_square_ratio: float,
    manual_grid: Optional[str] = None,
    grid_file: Optional[str] = None,
) -> List[List[str]]:
    """Generates crossword grid (various methods)."""

    if manual_grid:
        grid = generate_grid_from_string(manual_grid)
        if grid:
            return grid
        logging.warning("Invalid manual grid. Generating random grid instead.")

    if grid_file:
        grid = generate_grid_from_file(grid_file)
        if grid:
            return grid
        logging.warning("Invalid grid file. Generating random grid instead.")

    return generate_grid_random(width, height, black_square_ratio)


def find_slots(grid: List[List[str]]) -> List[Tuple[int, int, str, int]]:
    """Finds word slots (across and down)."""
    height = len(grid)
    width = len(grid[0])
    slots: List[Tuple[int, int, str, int]] = []
    with Progress() as progress:
        task = progress.add_task("[cyan]Finding Slots...", total=height + width)

        for row_index in range(height):
            start = -1
            for col_index in range(width):
                if grid[row_index][col_index] == ".":
                    if start == -1:
                        start = col_index
                elif start != -1:
                    length = col_index - start
                    if length > 1:
                        slots.append((row_index, start, "across", length))
                    start = -1
            if start != -1:
                length = width - start
                if length > 1:
                    slots.append((row_index, start, "across", length))
            progress.update(task, advance=1)

        for col_index in range(width):
            start = -1
            for row_index in range(height):
                if grid[row_index][col_index] == ".":
                    if start == -1:
                        start = row_index
                elif start != -1:
                    length = row_index - start
                    if length > 1:
                        slots.append((start, col_index, "down", length))
                    start = -1
            if start != -1:
                length = height - start
                if length > 1:
                    slots.append((start, col_index, "down", length))
            progress.update(task, advance=1)
    return slots


def load_words(filepath: str) -> Dict[int, List[str]]:
    """Loads and preprocesses words, groups by length."""
    words_by_length: Dict[int, List[str]] = {}
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            with Progress() as progress:
                file.seek(0, 2)  # Move to the end
                file_size = file.tell()  # Get file size
                file.seek(0)  # Back to beginning
                task = progress.add_task(
                    "[green]Loading and Preprocessing Words...", total=file_size
                )
                for line in file:
                    word = line.strip().upper()
                    word = WORD_CLEAN_RE.sub("", word)

                    if not word:
                        continue

                    length = len(word)
                    if length > 1:
                        if length not in words_by_length:
                            words_by_length[length] = []
                        words_by_length[length].append(word)
                    progress.update(task, advance=len(line.encode("utf-8")))

    except FileNotFoundError:
        logging.error(f"Word file not found at {filepath}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading words: {e}")
        sys.exit(1)

    return words_by_length


def is_valid_placement(
    grid: List[List[str]], word: str, row: int, col: int, direction: str
) -> bool:
    """Less restrictive placement validation."""
    if len(word) < MIN_WORD_LENGTH:
        return False
        
    length = len(word)
    height = len(grid)
    width = len(grid[0])
    
    # Check boundaries
    if direction == "across":
        if col + length > width:
            return False
    else:
        if row + length > height:
            return False
            
    # Check for valid intersections only
    if direction == "across":
        for i in range(length):
            current = grid[row][col + i]
            if current != "." and current != word[i]:
                return False
    else:
        for i in range(length):
            current = grid[row + i][col]
            if current != "." and current != word[i]:
                return False
                    
    return True


def place_word(
    grid: List[List[str]], word: str, row: int, col: int, direction: str
) -> List[List[str]]:
    """Places a word in the grid (creates a copy)."""
    new_grid = grid.copy()  # Create a shallow copy of the outer list
    for i in range(len(new_grid)):
        new_grid[i] = new_grid[i].copy() # Deep copy
    length = len(word)
    if direction == "across":
        for i in range(length):
            new_grid[row][col + i] = word[i]
    else:
        for i in range(length):
            new_grid[row + i][col] = word[i]
    return new_grid


def check_all_letters_connected(
    grid: List[List[str]], placed_words: List[Tuple[str, int, int, str]]
) -> bool:
    """Ensures all placed letters are part of across and down words."""
    height = len(grid)
    width = len(grid[0])
    letter_grid = [["" for _ in range(width)] for _ in range(height)]
    placed_coords = set()

    for word, row, col, direction in placed_words:
        for i, letter in enumerate(word):
            if direction == "across":
                letter_grid[row][col + i] = letter
                placed_coords.add((row, col + i))
            else:
                letter_grid[row + i][col] = letter
                placed_coords.add((row + i, col))

    for row, col in placed_coords:
        in_across = False
        in_down = False
        for word, word_row, word_col, word_dir in placed_words:
            if (
                word_dir == "across"
                and word_row == row
                and word_col <= col < word_col + len(word)
            ):
                in_across = True
            elif (
                word_dir == "down"
                and word_col == col
                and word_row <= row < word_row + len(word)
            ):
                in_down = True
        if not (in_across and in_down):
            return False  # Letter not in both

    return True  # All letters connected


def slots_intersect(
    slot1: Tuple[int, int, str, int], slot2: Tuple[int, int, str, int]
) -> bool:
    """Checks if two slots intersect."""
    row1, col1, dir1, len1 = slot1
    row2, col2, dir2, len2 = slot2

    if dir1 == dir2:
        return False  # Parallel

    if dir1 == "across":
        if col1 <= col2 < col1 + len1 and row2 <= row1 < row2 + len2:
            return True
    else:  # dir1 == "down"
        if row1 <= row2 < row1 + len1 and col2 <= col1 < col2 + len2:
            return True

    return False


def calculate_intersections(
    slots: List[Tuple[int, int, str, int]]
) -> Dict[int, List[int]]:
    """Calculates which slots intersect."""
    intersections: Dict[int, List[int]] = {i: [] for i in range(len(slots))}
    for i in range(len(slots)):
        for j in range(i + 1, len(slots)):
            if slots_intersect(slots[i], slots[j]):
                intersections[i].append(j)
                intersections[j].append(i)
    return intersections


def calculate_word_score(word: str) -> float:
    """Calculate word score based on letter frequencies and uniqueness."""
    # Common letters get lower scores to encourage more intersections
    common_letters = 'EARIOTNSLC'
    uncommon_letters = 'JQXZWYVKBP'
    
    score = 0.0
    for letter in word:
        if letter in common_letters:
            score += 0.5
        elif letter in uncommon_letters:
            score += 2.0
        else:
            score += 1.0
            
    # Bonus for words with diverse letters
    unique_letters = len(set(word))
    score += unique_letters * 0.5
    
    return score

def get_intersection_quality(
    word: str,
    row: int,
    col: int,
    direction: str,
    placed_words: List[Tuple[str, int, int, str]]
) -> float:
    """Calculate how well a word intersects with existing words."""
    score = 0.0
    length = len(word)
    
    for placed_word, p_row, p_col, p_dir in placed_words:
        if direction == "across" and p_dir == "down":
            if p_col >= col and p_col < col + length:
                if row >= p_row and row < p_row + len(placed_word):
                    # Reward intersections that use less common letters
                    intersection_letter = word[p_col - col]
                    if intersection_letter in 'JQXZW':
                        score += 3.0
                    elif intersection_letter in 'YVKBP':
                        score += 2.0
                    else:
                        score += 1.0
        elif direction == "down" and p_dir == "across":
            if p_row >= row and p_row < row + length:
                if col >= p_col and col < p_col + len(placed_word):
                    intersection_letter = word[p_row - row]
                    if intersection_letter in 'JQXZW':
                        score += 3.0
                    elif intersection_letter in 'YVKBP':
                        score += 2.0
                    else:
                        score += 1.0
    
    return score

class CrosswordStats:
    def __init__(self):
        self.attempts = 0
        self.backtracks = 0
        self.words_tried = 0
        self.successful_placements = 0
        self.failed_placements = 0
        self.time_spent = 0
        self.slot_fill_order = []
        
    def get_summary(self) -> str:
        return f"""
📊 Crossword Generation Stats:
├─ Total Attempts: {self.attempts}
├─ Backtracking Events: {self.backtracks}
├─ Words Evaluated: {self.words_tried}
├─ Successful Placements: {self.successful_placements}
├─ Failed Placements: {self.failed_placements}
├─ Success Rate: {(self.successful_placements/max(1,self.words_tried))*100:.1f}%
└─ Time Spent: {self.time_spent:.2f}s
"""

def try_slot_parallel(
    grid: List[List[str]],
    slot: Tuple[int, int, str, int],
    word: str,
    remaining_slots: List[Tuple[int, int, str, int]],
    words_by_length: Dict[int, List[str]],
    placed_words: List[Tuple[str, int, int, str]],
    max_attempts: int,
    start_time: float,
    timeout: int,
    progress: Progress,
    task: TaskID,
    stats: CrosswordStats,
    recursion_depth: int,
    cache: Dict[str, bool],
    beam_width: int
) -> Tuple[Optional[List[List[str]]], Optional[List[Tuple[str, int, int, str]]]]:
    """Helper function for parallel word placement attempts"""
    row, col, direction, _ = slot
    progress.update(task, advance=1)
    new_grid = place_word(grid, word, row, col, direction)
    new_placed_words = placed_words + [(word, row, col, direction)]
    stats.successful_placements += 1

    result = select_words_recursive(
        new_grid,
        remaining_slots,
        words_by_length,
        new_placed_words,
        max_attempts,
        start_time,
        timeout,
        progress,
        task,
        stats,
        recursion_depth + 1,
        cache,
        beam_width
    )
    
    return result

def select_words_recursive(
    grid: List[List[str]],
    slots: List[Tuple[int, int, str, int]],
    words_by_length: Dict[int, List[str]], 
    placed_words: List[Tuple[str, int, int, str]],
    max_attempts: int,
    start_time: float,
    timeout: int,
    progress: Progress,
    task: TaskID,
    stats: CrosswordStats,
    recursion_depth: int = 0,
    cache: Dict[str, bool] = None,
    beam_width: int = DEFAULT_BEAM_WIDTH,
) -> Tuple[Optional[List[List[str]]], Optional[List[Tuple[str, int, int, str]]]]:
    """Enhanced recursive word selection with parallel processing."""
    cache = cache or {}
    
    stats.attempts += 1
    current_time = time.time()
    stats.time_spent = current_time - start_time

    # Early termination checks
    if current_time - start_time > timeout or recursion_depth > MAX_RECURSION_DEPTH:
        stats.backtracks += 1
        return None, None

    if not slots:
        if check_all_letters_connected(grid, placed_words):
            print("\nSuccessful grid configuration:")
            print_grid(grid)
            return grid, placed_words
        return None, None

    # Enhanced slot selection with scoring
    slot_scores = []
    for slot in slots:
        row, col, direction, length = slot
        
        # Calculate constrainedness score
        constraints = 0
        possible_words = 0
        for word in words_by_length.get(length, []):
            if _is_valid_cached(grid, word, row, col, direction, cache):
                possible_words += 1
                constraints += sum(1 for c in word if c in "JQXZW")
        
        if possible_words == 0:
            stats.failed_placements += 1
            return None, None
            
        score = (1.0/possible_words) * (constraints + 1)
        slot_scores.append((score, slot))
    
    slot_scores.sort(reverse=True)
    
    # Process most constrained slots first
    for _, slot in slot_scores[:beam_width]:
        row, col, direction, length = slot
        remaining_slots = [s for s in slots if s != slot]

        # Score candidate words
        word_scores = []
        for word in words_by_length.get(length, []):
            if _is_valid_cached(grid, word, row, col, direction, cache):
                score = calculate_word_score(word) + get_intersection_quality(
                    word, row, col, direction, placed_words
                )
                word_scores.append((score, word))
        
        word_scores.sort(reverse=True)
        
        # Try best words in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for _, word in word_scores[:DEFAULT_MAX_BACKTRACK]:
                futures.append(
                    executor.submit(
                        try_slot_parallel,
                        grid,
                        slot,
                        word,
                        remaining_slots,
                        words_by_length,
                        placed_words,
                        max_attempts,
                        start_time,
                        timeout,
                        progress,
                        task,
                        stats,
                        recursion_depth,
                        cache,
                        beam_width
                    )
                )
            
            # Process results as they complete
            for future in as_completed(futures):
                result = future.result()
                if result[0] is not None:
                    return result

    stats.backtracks += 1
    return None, None

def _is_valid_cached(
    grid: List[List[str]], 
    word: str,
    row: int,
    col: int,
    direction: str,
    cache: Dict[str, bool]
) -> bool:
    """Cached version of is_valid_placement."""
    key = f"{word}:{row}:{col}:{direction}"
    if key not in cache:
        cache[key] = is_valid_placement(grid, word, row, col, direction)
    return cache[key]

def _validate_remaining_slots(
    grid: List[List[str]],
    slots: List[Tuple[int, int, str, int]],
    words_by_length: Dict[int, List[str]],
    cache: Dict[str, bool]
) -> bool:
    """Forward checking - ensure all slots have at least one valid word."""
    for row, col, direction, length in slots:
        if not any(_is_valid_cached(grid, word, row, col, direction, cache)
                  for word in words_by_length.get(length, [])):
            return False
    return True

def _words_intersect(
    word1: Tuple[str, int, int, str],
    word2: Tuple[str, int, int, str]
) -> bool:
    """Check if two word placements intersect."""
    _, row1, col1, dir1 = word1
    _, row2, col2, dir2 = word2
    
    if dir1 == dir2:
        return False
        
    if dir1 == "across":
        return (col2 >= col1 and col2 < col1 + len(word1[0]) and
                row1 >= row2 and row1 < row2 + len(word2[0]))
    else:
        return (row2 >= row1 and row2 < row1 + len(word1[0]) and
                col1 >= col2 and col1 < col2 + len(word2[0]))


def select_words(
    grid: List[List[str]],
    slots: List[Tuple[int, int, str, int]],
    words_by_length: Dict[int, List[str]],
    progress: Progress,
    task: TaskID,
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[Optional[List[List[str]]], Optional[List[Tuple[str, int, int, str]]]]:
    """Enhanced word selection with statistics tracking."""
    
    stats = CrosswordStats()
    start_time = time.time()
    
    intersections = calculate_intersections(slots)
    sorted_slots = sorted(
        slots,
        key=lambda x: (len(intersections.get(slots.index(x), [])), x[3]),
        reverse=True,
    )

    total_estimated_attempts = sum(
        sum(1 for word in words_by_length.get(slot[3], [])
            if is_valid_placement(grid, word, slot[0], slot[1], slot[2]))
        for slot in sorted_slots
    )
    progress.update(task, total=total_estimated_attempts)
    
    filled_grid, placed_words = select_words_recursive(
        grid,
        sorted_slots,
        words_by_length,
        [],
        DEFAULT_MAX_ATTEMPTS,
        start_time,
        timeout,
        progress,
        task,
        stats
    )

    # Print statistics
    logging.info(stats.get_summary())
    
    return filled_grid, placed_words


def print_grid(grid: List[List[str]]) -> None:
    """Prints a visual representation of the grid."""
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    table = Table(show_header=False, show_edge=False, padding=0)
    
    for row in grid:
        table.add_row(*[
            "[white on black]  [/]" if cell == "#" else "[black on white]  [/]" 
            for cell in row
        ])
    
    console.print(table)


def main():
    """Main function to parse arguments and run crossword generation."""
    parser = argparse.ArgumentParser(
        description="Generates an interactive crossword puzzle."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_GRID_WIDTH,
        help="Width of grid (columns).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_GRID_HEIGHT,
        help="Height of grid (rows).",
    )
    parser.add_argument(
        "--black_squares",
        type=float,
        default=DEFAULT_BLACK_SQUARE_RATIO,
        help="Approximate % of black squares (0.0 to 1.0).",
    )
    parser.add_argument(
        "--manual_grid",
        type=str,
        default=None,
        help="Manually specify grid ('.'=white, '#'=black).",
    )
    parser.add_argument(
        "--grid_file",
        type=str,
        default=None,
        help="Path to file with grid layout.",
    )
    parser.add_argument(
        "--lm_studio_url",
        type=str,
        default=DEFAULT_LM_STUDIO_URL,
        help="LM Studio server URL.",
    )
    parser.add_argument(
        "--words_file",
        type=str,
        default=DEFAULT_WORDS_FILE,
        help="Path to words file (one word per line).",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=DEFAULT_OUTPUT_FILENAME,
        help="Output HTML filename.",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="Max attempts to place a word.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Timeout for word selection (seconds).",
    )
    parser.add_argument(
        "--llm_timeout",
        type=int,
        default=DEFAULT_LLM_TIMEOUT,
        help="Timeout for LLM requests (seconds).",
    )
    parser.add_argument(
        "--llm_max_tokens",
        type=int,
        default=DEFAULT_LLM_MAX_TOKENS,
        help="Max tokens for LLM responses.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=DEFAULT_LANGUAGE,
        help="Language for definitions.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name to use for definitions.",
    )
    parser.add_argument(
        "--max_grid_iterations",
        type=int,
        default=DEFAULT_MAX_GRID_ITERATIONS,
        help="Maximum number of attempts to generate a complete grid.",
    )

    args = parser.parse_args()

     # Input validation
    if args.width <= 0 or args.height <= 0:
        logging.error("Width and height must be positive integers.")
        sys.exit(1)
    if not 0.0 <= args.black_squares <= 1.0:
        logging.error("black_squares must be between 0.0 and 1.0.")
        sys.exit(1)
    if args.max_attempts <=0 or args.timeout <= 0 or args.llm_timeout <= 0 or args.llm_max_tokens <= 0:
        logging.error("All timeout and max_attempts values need to be positive integers")
        sys.exit(1)
    if args.max_grid_iterations <= 0:
        logging.error("max_grid_iterations must be positive")
        sys.exit(1)

    llm = setup_langchain_llm(
        args.lm_studio_url, 
        args.llm_timeout, 
        args.llm_max_tokens,
        args.model
    )

    words_by_length = load_words(args.words_file)
    if not words_by_length:
        logging.error("No valid words found in the word file.")
        sys.exit(1)

    # Try multiple times to generate a valid grid
    filled_grid = None
    placed_words = None
    iteration = 0

    while filled_grid is None and iteration < args.max_grid_iterations:
        iteration += 1
        logging.info(f"\nAttempt {iteration} to generate grid...")
        
        grid = generate_grid(
            args.width, args.height, args.black_squares, 
            args.manual_grid, args.grid_file
        )
        
        print("\nInitial grid configuration:")
        print_grid(grid)
        
        slots = find_slots(grid)
        if len(slots) < 2:
            logging.warning("Grid has insufficient slots, retrying...")
            continue

        # Check for impossible grids early
        impossible = False
        for _, _, _, length in slots:
            if length not in words_by_length:
                logging.warning(f"No words of length {length} found, retrying...")
                impossible = True
                break
        if impossible:
            continue

        with Progress() as progress:
            task_select_words = progress.add_task("[yellow]Selecting Words...")
            filled_grid, placed_words = select_words(
                grid, slots, words_by_length, 
                progress, task_select_words, args.timeout
            )
            progress.stop()

    if filled_grid is not None:
        definitions = generate_definitions(placed_words, llm, args.language)
        create_html(filled_grid, placed_words, definitions, args.output_filename)
        print(f"Crossword puzzle created and saved to {args.output_filename}")
    else:
        print("Failed to generate a complete crossword puzzle.")


if __name__ == "__main__":
    main()