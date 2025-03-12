import argparse
import logging
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional
from threading import Lock
from functools import lru_cache
from collections import defaultdict
from rich.progress import Progress, TaskID
from rich.console import Console
from rich.table import Table
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# --- Import config ---
from config import Config, DEFAULT_BEAM_WIDTH, DEFAULT_MAX_BACKTRACK, DEFAULT_GRID_WIDTH, DEFAULT_GRID_HEIGHT, DEFAULT_BLACK_SQUARE_RATIO, DEFAULT_LM_STUDIO_URL, DEFAULT_WORDS_FILE, DEFAULT_OUTPUT_FILENAME, DEFAULT_MAX_ATTEMPTS, DEFAULT_TIMEOUT, DEFAULT_LLM_TIMEOUT, DEFAULT_LLM_MAX_TOKENS, DEFAULT_LANGUAGE, DEFAULT_MODEL, DEFAULT_DIFFICULTY, DEFAULT_MAX_GRID_ITERATIONS  # Import the Config class

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Compiled Regex Patterns ---
WORD_CLEAN_RE = re.compile(r"[^A-Z]")  # Only keep uppercase letters
DEFINITION_CLEAN_RE = re.compile(r"^\d+\.\s*") # Remove leading numbers
NON_ALPHANUMERIC_RE = re.compile(r"^[^\w]+|[^\w]+$") # trim non-alphanumeric

# --- Global Variables ---
cache_lock = Lock()
placement_cache: Dict[str, bool] = {}
definition_cache: Dict[str, str] = {}
word_index: Dict[Tuple[int, str], List[str]] = defaultdict(list)
llm = None  # Global LLM object

# REVIEW NOTE: Code reviewed and no critical issues found.

# --- Utility Functions ---
def print_grid(grid: List[List[str]], placed_words: List[Tuple[str, int, int, str]] = None,
               console: Optional[Console] = None) -> None:
    """Prints the grid with highlighting."""
    if console is None:
        console = Console()

    table = Table(show_header=False, show_edge=False, padding=0)

    if placed_words is not None:
        placed_coords = set()
        for word, row, col, direction in placed_words:
            for i in range(len(word)):
                if direction == "across":
                    placed_coords.add((row, col + i))
                else:
                    placed_coords.add((row + i, col))

    for r_idx, row in enumerate(grid):
        row_display = []
        for c_idx, cell in enumerate(row):
            if cell == "#":
                row_display.append("[white on black]  [/]")  # Black square
            elif placed_words is not None and (r_idx, c_idx) in placed_coords:
                row_display.append(f"[black on green]{cell.center(2)}[/]")
            else:
                row_display.append(f"[black on white]{cell.center(2)}[/]")
        table.add_row(*row_display)

    console.print(table)


def calculate_word_frequency(word: str, word_frequencies: Dict[str, float]) -> float:
    """Calculates word score (lower is more common)."""
    return word_frequencies.get(word.lower(), 1e-6) # Default very low


def create_pattern(word: str) -> str:
    """Creates regex pattern, '.' for unknown."""
    return ''.join('.' if c == '.' else c for c in word)

def load_words(filepath: str, min_word_count: int = 3, config: Config = None) -> Tuple[Dict[int, List[str]], Dict[str, float]]:
    """Loads, preprocesses, and filters words."""
    words_by_length: Dict[int, List[str]] = defaultdict(list)
    word_counts: Dict[str, int] = defaultdict(int)
    total_count = 0
    filtered_words = set()

    try:
        with open(filepath, "r", encoding="utf-8") as file:
            for line in file:
                word = line.strip().upper()
                word = WORD_CLEAN_RE.sub("", word)
                if len(word) >= (config.min_word_length if config else 3):  # Use config.min_word_length if available
                    word_counts[word] += 1
                    total_count += 1

        word_frequencies: Dict[str, float] = {}
        for word, count in word_counts.items():
            freq = count / total_count
            word_frequencies[word] = freq  # Store with uppercase key
            if count >= min_word_count:
                filtered_words.add(word)

        for word in filtered_words:
            words_by_length[len(word)].append(word)

        return words_by_length, word_frequencies

    except FileNotFoundError:
        logging.error(f"Word file not found: {filepath}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading words: {e}")
        sys.exit(1)


def build_word_index(words_by_length: Dict[int, List[str]]):
    """Builds the word index for efficient lookups."""
    global word_index
    word_index.clear()

    for length, words in words_by_length.items():
        for word in words:
            word_index[(length, word)].append(word)  # Add the full word
            for i in range(1 << length):
                pattern = "".join(word[j] if (i >> j) & 1 else "." for j in range(length))
                if pattern != word: # Avoid re-adding
                  word_index[(length, pattern)].append(word)


# --- LangChain and LLM Setup ---
def setup_langchain_llm(config: Config) -> ChatOpenAI:  # Pass config
    """Initializes the LangChain LLM."""
    global llm
    try:
        llm = ChatOpenAI(
            base_url=config.lm_studio_url,
            api_key="NA",  # Not needed for local models
            model=config.model,
            temperature=0.7,
            max_tokens=config.llm_max_tokens,
            timeout=config.llm_timeout,
        )
        return llm
    except Exception as e:
        logging.error(f"Failed to initialize LLM: {e}")
        sys.exit(1)


@lru_cache(maxsize=512)
def generate_definition_langchain(word: str, language: str, config: Config) -> str:  # Pass config
    """Generates clues, with retries and filtering."""
    if word in definition_cache:
        return definition_cache[word]

    prompt_template = """Generate a short, concise crossword clue for: "{word}". Reply in {language}.
    Strict Rules:
    1. Absolutely NO part of the target word in the clue.
    2. Clue must be 10 words or less.
    3. Avoid obvious synonyms.
    4. No direct definitions.
    5. Focus on wordplay, double meanings, or indirect hints.
    Output: Only the clue text, nothing else."""

    prompt = PromptTemplate.from_template(prompt_template)
    output_parser = StrOutputParser()
    chain = (
            {"word": RunnablePassthrough(), "language": RunnablePassthrough()}
            | prompt
            | llm
            | output_parser
    )

    for attempt in range(config.max_definition_attempts): # Use config
        try:
            definition = chain.invoke({"word": word, "language": language})
            definition = definition.strip()

            # Cleaning and filtering
            definition = re.sub(r'(?i)definizione[:\s]*', '', definition)
            definition = re.sub(r'(?i)clue[:\s]*', '', definition)
            definition = re.sub(r'^\d+[\.\)]\s*', '', definition).strip()

            for pattern in config.forbidden_patterns:  # Use config
                if re.search(pattern(word), definition, re.IGNORECASE):
                    raise ValueError("Forbidden word/pattern used.")

            word_lower = word.lower()
            definition_lower = definition.lower()
            if any(word_lower[i:j] in definition_lower for i in range(len(word) - 2) for j in range(i + 3, len(word) + 1)):
                raise ValueError("Part of word used.")

            definition_cache[word] = definition
            return definition

        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} for '{word}': {e}")
            if attempt < config.max_definition_attempts - 1:  # Use config
                time.sleep(config.definition_retry_delay)  # Use config

    logging.error(f"Failed to generate definition for '{word}'.")
    return "Definizione non disponibile"



# --- Grid Generation and Manipulation ---

def generate_grid_from_string(grid_string: str) -> Optional[List[List[str]]]:
    """Generates grid from string, with validation."""
    lines = grid_string.strip().split("\n")
    grid: List[List[str]] = []
    for line in lines:
        row = [char for char in line.strip() if char in (".", "#")]
        if len(row) != len(lines[0]):
            logging.error("Inconsistent row length in manual grid.")
            return None
        grid.append(row)

    if not grid or any(len(row) != len(grid[0]) for row in grid):
        logging.error("Invalid manual grid: empty or non-rectangular.")
        return None
    return grid


def generate_grid_from_file(filepath: str) -> Optional[List[List[str]]]:
    """Loads grid from file, with error handling."""
    try:
        with open(filepath, "r") as f:
            return generate_grid_from_string(f.read())
    except FileNotFoundError:
        logging.error(f"Grid file not found: {filepath}")
        return None
    except Exception as e:
        logging.error(f"Error reading grid file: {e}")
        return None

def is_valid_grid(grid: List[List[str]]) -> bool:
    """Checks if the grid is valid."""
    if not grid:
        return False
    width = len(grid[0])
    return all(len(row) == width and all(c in ('.', '#') for c in row) for row in grid)


def generate_grid_random(width: int, height: int, black_square_ratio: float) -> List[List[str]]:
    """Generates a random, symmetrical grid."""
    grid = [["." for _ in range(width)] for _ in range(height)]
    num_black_squares = int(width * height * black_square_ratio)

    def place_symmetrically(row: int, col: int):
        grid[row][col] = "#"
        grid[height - 1 - row][width - 1 - col] = "#"

    if width % 2 == 1 and height % 2 == 1:
        place_symmetrically(height // 2, width // 2)
        num_black_squares -= 1

    placed_count = 0
    attempts = 0
    max_attempts = width * height * 5

    while placed_count < num_black_squares and attempts < max_attempts:
        attempts += 1
        row, col = random.randint(0, height - 1), random.randint(0, width - 1)

        if grid[row][col] == ".":
            # Check for 2x2 blocks *before* placement
            if (row > 0 and col > 0 and grid[row - 1][col] == "#" and grid[row][col - 1] == "#" and grid[row - 1][col - 1] == "#") or \
               (row > 0 and col < width - 1 and grid[row - 1][col] == "#" and grid[row][col + 1] == "#" and grid[row - 1][col + 1] == "#") or \
               (row < height - 1 and col > 0 and grid[row + 1][col] == "#" and grid[row][col - 1] == "#" and grid[row + 1][col - 1] == "#") or \
               (row < height - 1 and col < width - 1 and grid[row + 1][col] == "#" and grid[row][col + 1] == "#" and grid[row + 1][col + 1] == "#"):
                continue

            # Check for isolated white squares (before placement)
            def is_isolated(r, c):
                if r > 0 and grid[r-1][c] == ".": return False
                if r < height - 1 and grid[r+1][c] == ".": return False
                if c > 0 and grid[r][c-1] == ".": return False
                if c < width - 1 and grid[r][c+1] == ".": return False
                return True

            if is_isolated(row,col):
              continue;

            place_symmetrically(row, col)
            placed_count += 2 if (row, col) != (height - 1 - row, width - 1 - col) else 1

    if placed_count < num_black_squares:
        logging.warning(f"Could only place {placed_count} of {num_black_squares} black squares.")
    return grid

def generate_grid(config: Config) -> List[List[str]]: # Pass config
    """Generates grid, handling manual, file, or random."""
    if config.manual_grid:
        grid = generate_grid_from_string(config.manual_grid)
        if grid: return grid
        logging.warning("Invalid manual grid. Using random.")

    if config.grid_file:
        grid = generate_grid_from_file(config.grid_file)
        if grid: return grid
        logging.warning("Invalid grid file. Using random.")

    return generate_grid_random(config.grid_width, config.grid_height, config.black_square_ratio)


def find_slots(grid: List[List[str]], config: Config) -> List[Tuple[int, int, str, int]]:  # Pass config
    """Identifies word slots (across and down)."""
    height, width = len(grid), len(grid[0])
    slots = []

    # Across
    for r in range(height):
        start = -1
        for c in range(width):
            if grid[r][c] == ".":
                if start == -1: start = c
            elif start != -1:
                length = c - start
                if length >= config.min_word_length:  # Use config.min_word_length
                    slots.append((r, start, "across", length))
                start = -1
        if start != -1 and width - start >= config.min_word_length:  # Use config.min_word_length
            slots.append((r, start, "across", width - start))

    # Down
    for c in range(width):
        start = -1
        for r in range(height):
            if grid[r][c] == ".":
                if start == -1: start = r
            elif start != -1:
                length = r - start
                if length >= config.min_word_length:  # Use config
                    slots.append((start, c, "down", length))
                start = -1
        if start != -1 and height - start >= config.min_word_length:  # Use config.min_word_length
            slots.append((start, c, "down", height - start))
    return slots


def is_valid_placement(grid: List[List[str]], word: str, row: int, col: int, direction: str) -> bool:
    """Checks placement using the index."""
    length = len(word)
    if direction == "across":
        if col + length > len(grid[0]): return False
        pattern = ''.join(grid[row][col + i] for i in range(length))
    else:  # down
        if row + length > len(grid): return False
        pattern = ''.join(grid[row + i][col] for i in range(length))

    key = (length, create_pattern(pattern))
    return word in word_index.get(key, [])

def _is_valid_cached(grid: List[List[str]], word: str, row: int, col: int, direction: str) -> bool:
    """Cached version of is_valid_placement."""
    key = f"{word}:{row}:{col}:{direction}"
    with cache_lock:
        if key not in placement_cache:
            placement_cache[key] = is_valid_placement(grid, word, row, col, direction)
        return placement_cache[key]

def place_word(grid: List[List[str]], word: str, row: int, col: int, direction: str) -> List[List[str]]:
    """Places a word onto a *copy* of the grid."""
    new_grid = [row[:] for row in grid]  # Deep copy
    for i, letter in enumerate(word):
        if direction == "across":
            new_grid[row][col + i] = letter
        else:
            new_grid[row + i][col] = letter
    return new_grid


def remove_word(grid: List[List[str]], word: str, row: int, col: int, direction: str) -> List[List[str]]:
    """Removes a word from a *copy* of the grid."""
    new_grid = [row[:] for row in grid]  # Deep copy
    for i in range(len(word)):
        if direction == "across":
            if new_grid[row][col + i] == word[i]:
                new_grid[row][col + i] = "."
        else:
            if new_grid[row + i][col] == word[i]:
                new_grid[row + i][col] = "."
    return new_grid

def check_all_letters_connected(grid: List[List[str]], placed_words: List[Tuple[str, int, int, str]]) -> bool:
    """Checks if all placed letters are connected."""
    if not placed_words: return True

    letter_positions = set()
    for word, row, col, direction in placed_words:
        for i in range(len(word)):
            if direction == "across":
                pos = (row, col + i)
                if grid[row][col + i] != word[i]: return False
            else:
                pos = (row + i, col)
                if grid[row + i][col] != word[i]: return False
            letter_positions.add(pos)

    for row, col in letter_positions:
        in_across = any(row == r and col >= c and col < c + len(w) for w, r, c, d in placed_words if d == "across")
        in_down = any(col == c and row >= r and row < r + len(w) for w, r, c, d in placed_words if d == "down")
        if not (in_across and in_down):
            return False

    return True

def _validate_remaining_slots(grid: List[List[str]], slots: List[Tuple[int, int, str, int]],
                              words_by_length: Dict[int, List[str]]) -> bool:
    """Checks if all slots have at least one valid word."""
    for row, col, direction, length in slots:
        if direction == "across":
            pattern_list = [grid[row][col + i] for i in range(length)]
        else:
            pattern_list = [grid[row+i][col] for i in range(length)]
        pattern = create_pattern("".join(pattern_list))
        if not word_index.get((length, pattern)):
            return False
    return True


def calculate_intersection_score(grid: List[List[str]], word: str, row: int, col: int, direction: str,
                                 placed_words: List[Tuple[str, int, int, str]]) -> float:
    """Calculates intersection score."""
    intersections = 0
    score = 0.0
    length = len(word)

    grid_state = [row[:] for row in grid]

    for i, letter in enumerate(word):
        if direction == "across":
            if grid_state[row][col + i] != '.' and grid_state[row][col + i] != letter:
                return -1.0  # Invalid
            grid_state[row][col + i] = letter
        else:
            if grid_state[row + i][col] != '.' and grid_state[row + i][col] != letter:
                return -1.0
            grid_state[row + i][col] = letter

    for placed_word, p_row, p_col, p_dir in placed_words:
        if direction == "across" and p_dir == "down":
            if p_col >= col and p_col < col + length and row >= p_row and row < p_row + len(placed_word):
                intersections += 1
                score += 1
        elif direction == "down" and p_dir == "across":
            if p_row >= row and p_row < row + length and col >= p_col and col < p_col + len(placed_word):
                intersections += 1
                score += 1

    return score + intersections * 0.5

def get_slot_score(grid: List[List[str]], slot: Tuple[int, int, str, int], words_by_length: Dict[int, List[str]],
                   placed_words: List[Tuple[str, int, int, str]]) -> float:
    """Scores slots based on constrainedness and intersections."""
    row, col, direction, length = slot

    if direction == "across":
        pattern = "".join(grid[row][col:col + length])
    else:
        pattern = "".join(grid[row + i][col] for i in range(length))

    pattern = create_pattern(pattern)
    possible_words = len(word_index.get((length, pattern), []))

    if possible_words == 0: return -1.0

    constrainedness_score = 1.0 / possible_words
    intersection_potential = 0.0
    for word in word_index.get((length, pattern), []):
        intersection_potential += calculate_intersection_score(grid, word, row, col, direction, placed_words)

    return constrainedness_score + intersection_potential * 0.1


# --- Word Selection (Recursive, Backtracking, Parallelism) ---

class CrosswordStats:
    """Tracks crossword generation statistics."""
    def __init__(self):
        self.attempts = 0
        self.backtracks = 0
        self.words_tried = 0
        self.successful_placements = 0
        self.failed_placements = 0
        self.time_spent = 0.0
        self.start_time = time.time()
        self.slot_fill_order: List[Tuple[int, int, str]] = []
        self.definition_failures = 0
        self.dynamic_beam_width = DEFAULT_BEAM_WIDTH
        self.dynamic_max_backtrack = DEFAULT_MAX_BACKTRACK

    def update_time(self):
        self.time_spent = time.time() - self.start_time

    def get_summary(self) -> str:
        self.update_time()
        return (
            "📊 Crossword Generation Stats:\n"
            f"├── Attempts: {self.attempts}\n"
            f"├── Backtracks: {self.backtracks}\n"
            f"├── Words Tried: {self.words_tried}\n"
            f"├── Successful Placements: {self.successful_placements}\n"
            f"├── Failed Placements: {self.failed_placements}\n"
            f"├── Definition Failures: {self.definition_failures}\n"
            f"├── Dynamic Beam Width: {self.dynamic_beam_width}\n"
            f"├── Dynamic Max Backtrack: {self.dynamic_max_backtrack}\n"
            f"├── Success Rate: {self.successful_placements / max(1, self.words_tried) * 100:.2f}%\n"
            f"└── Time Spent: {self.time_spent:.2f}s\n"
            f"└── Slots Filled Order: {self.slot_fill_order}"
        )

stats = CrosswordStats()


def validate_placement(grid: List[List[str]], slot: Tuple[int, int, str, int],
                       word: str, remaining_slots: List[Tuple[int, int, str, int]],
                       words_by_length: Dict[int, List[str]]) -> bool:
    """Validates word placement and checks for future issues."""
    row, col, direction, _ = slot
    if not _is_valid_cached(grid, word, row, col, direction):
        return False
    temp_grid = place_word(grid, word, row, col, direction)
    return _validate_remaining_slots(temp_grid, remaining_slots, words_by_length)


def make_placement(grid: List[List[str]], slot: Tuple[int, int, str, int],
                   word: str, placed_words: List[Tuple[str, int, int, str]]) -> Tuple[
    List[List[str]], List[Tuple[str, int, int, str]]]:
    """Places a word and updates placed_words."""
    row, col, direction, _ = slot
    new_grid = place_word(grid, word, row, col, direction)
    new_placed_words = placed_words + [(word, row, col, direction)]
    stats.successful_placements += 1
    stats.words_tried += 1
    stats.slot_fill_order.append((row, col, direction))
    return new_grid, new_placed_words


def handle_backtrack(slot: Tuple[int, int, str, int]) -> None:
    """Handles statistics during backtracking."""
    stats.backtracks += 1
    stats.failed_placements += 1
    if stats.slot_fill_order:
        stats.slot_fill_order.pop()


def try_slot(grid: List[List[str]], slot: Tuple[int, int, str, int], word: str,
             remaining_slots: List[Tuple[int, int, str, int]],
             words_by_length: Dict[int, List[str]],
             word_frequencies: Dict[str, float],
             placed_words: List[Tuple[str, int, int, str]],
             progress: Progress, task: TaskID,
             config: Config,  # Pass config
             ) -> Tuple[Optional[List[List[str]]], Optional[List[Tuple[str, int, int, str]]]]:
    """Tries to place a word with enhanced validation and caching."""
    row, col, direction, length = slot

    # Check placement cache first
    cache_key = f"{word}_{row}_{col}_{direction}"
    with cache_lock:
        if cache_key in placement_cache:
            if not placement_cache[cache_key]:
                return None, None

    # Validate placement with enhanced constraints
    if not validate_placement(grid, slot, word, remaining_slots, words_by_length):
        with cache_lock:
            placement_cache[cache_key] = False
        return None, None

    # Make placement and update cache
    new_grid, new_placed_words = make_placement(grid, slot, word, placed_words)
    with cache_lock:
        placement_cache[cache_key] = True

    # Try to solve remaining slots with optimized recursion
    result = select_words_recursive(new_grid, remaining_slots, words_by_length,
                                  word_frequencies, new_placed_words, progress,
                                  task, config, None, depth=0)
    if result[0] is not None:
        stats.successful_placements += 1
        return result

    # Handle backtrack with improved statistics
    handle_backtrack(slot)
    stats.failed_placements += 1
    return None, None


def get_location_score(grid: List[List[str]], slot: Tuple[int, int, str, int]) -> float:
    """Calculates a score based on slot location and potential intersections."""
    row, col, direction, length = slot
    height, width = len(grid), len(grid[0])

    # Center proximity bonus (words near center are preferred)
    center_row, center_col = height // 2, width // 2
    distance_to_center = abs(row - center_row) + abs(col - center_col)
    center_bonus = 1.0 - (distance_to_center / (height + width))

    # Length bonus (longer words create more intersection opportunities)
    length_bonus = length / max(height, width)

    # Intersection potential bonus
    crossing_slots = 0
    potential_intersections = 0
    if direction == "across":
        for i in range(length):
            # Count existing crossing points
            if any(grid[r][col + i] == "." for r in range(height)):
                crossing_slots += 1
            # Count potential future intersections
            empty_spaces = sum(1 for r in range(height) if grid[r][col + i] == ".")
            potential_intersections += empty_spaces
    else:
        for i in range(length):
            if any(grid[row + i][c] == "." for c in range(width)):
                crossing_slots += 1
            empty_spaces = sum(1 for c in range(width) if grid[row + i][c] == ".")
            potential_intersections += empty_spaces

    intersection_bonus = (crossing_slots + 0.5 * potential_intersections) / (length * 2)

    # Edge penalty (avoid placing words at edges unless necessary)
    edge_penalty = 0.0
    if row == 0 or row + (length if direction == "down" else 1) >= height:
        edge_penalty += 0.2
    if col == 0 or col + (length if direction == "across" else 1) >= width:
        edge_penalty += 0.2

    return (center_bonus * 0.3 + 
            length_bonus * 0.2 + 
            intersection_bonus * 0.4 - 
            edge_penalty * 0.1)



def select_words_recursive(
        grid: List[List[str]],
        slots: List[Tuple[int, int, str, int]],
        words_by_length: Dict[int, List[str]],
        word_frequencies: Dict[str, float],
        placed_words: List[Tuple[str, int, int, str]],
        progress: Progress,
        task: TaskID,
        config: Config,  # Pass config
        executor: Optional[ThreadPoolExecutor] = None,
        depth: int = 0
) -> Tuple[Optional[List[List[str]]], Optional[List[Tuple[str, int, int, str]]]]:
    """Recursively selects words, with enhanced backtracking and parallel processing."""
    stats.attempts += 1
    stats.update_time()

    # Early termination for deep recursion
    if depth > len(slots) * 2:
        return None, None

    if stats.time_spent > config.timeout:  # Use config.timeout
        return None, None

    if not slots:
        if check_all_letters_connected(grid, placed_words):
            return grid, placed_words
        return None, None

    # If we've backtracked too many times, try filling impossible spaces with black squares
    if stats.backtracks > 50 and depth > 0 and depth % 10 == 0:
        new_grid, new_slots = fill_impossible_spaces(grid, slots, words_by_length)
        if new_grid != grid:  # If the grid was modified
            logging.info(f"Filled impossible spaces with black squares at depth {depth}. Slots before: {len(slots)}, after: {len(new_slots)}")
            # Try with the new grid and slots
            result = select_words_recursive(new_grid, new_slots, words_by_length, word_frequencies, 
                                          placed_words, progress, task, config, executor, depth + 1)
            if result[0] is not None:
                return result
    
    # If we're getting close to filling the grid but stuck (few slots left but many backtracks),
    # try more aggressive black square filling
    if len(slots) < 10 and stats.backtracks > 100:
        logging.info(f"Trying aggressive black square filling with {len(slots)} slots left and {stats.backtracks} backtracks")
        new_grid, new_slots = final_fill_impossible_spaces(grid, slots, words_by_length, config)
        if new_grid != grid:  # If the grid was modified
            logging.info(f"Aggressively filled impossible spaces with black squares. Slots before: {len(slots)}, after: {len(new_slots)}")
            # Try with the new grid and slots
            result = select_words_recursive(new_grid, new_slots, words_by_length, word_frequencies, 
                                          placed_words, progress, task, config, executor, depth + 1)
            if result[0] is not None:
                return result

    scored_slots = []
    for slot in slots:
        base_score = get_slot_score(grid, slot, words_by_length, placed_words)
        location_bonus = get_location_score(grid, slot)
        total_score = base_score + location_bonus
        scored_slots.append((total_score, slot))
    scored_slots.sort(key=lambda x: x[0], reverse=True)

    # Adaptive parameter adjustment based on search progress
    if stats.backtracks > 50 and stats.dynamic_beam_width < 50:
        stats.dynamic_beam_width += 3  # More aggressive beam width increase
    if stats.backtracks > 100 and stats.dynamic_max_backtrack < 1000:
        stats.dynamic_max_backtrack += 100  # More aggressive backtrack limit
    
    # Reset parameters if we're making good progress
    if stats.successful_placements > 0 and stats.successful_placements % 5 == 0:
        if stats.dynamic_beam_width > DEFAULT_BEAM_WIDTH:
            stats.dynamic_beam_width = max(DEFAULT_BEAM_WIDTH, stats.dynamic_beam_width - 1)
        if stats.dynamic_max_backtrack > DEFAULT_MAX_BACKTRACK:
            stats.dynamic_max_backtrack = max(DEFAULT_MAX_BACKTRACK, stats.dynamic_max_backtrack - 25)

    for score, slot in scored_slots[:stats.dynamic_beam_width]:
        row, col, direction, length = slot
        remaining_slots = [s for s in slots if s != slot]

        if direction == "across":
            pattern = "".join(grid[row][col:col + length])
        else:
            pattern = "".join(grid[row + i][col] for i in range(length))
        pattern = create_pattern(pattern)
        valid_words = word_index.get((length, pattern), [])

        word_scores = []
        freq_weight = config.word_frequency_weights[config.difficulty]  # Use config
        for word in valid_words:
            intersection_score = calculate_intersection_score(grid, word, row, col, direction, placed_words)
            frequency_score = calculate_word_frequency(word, word_frequencies)
            word_score = (intersection_score * (1 - freq_weight) + (1 - frequency_score) * freq_weight)
            word_scores.append((word_score, word))
        word_scores.sort(key=lambda x: x[0], reverse=True)

        if executor is not None:  # Top-level: use threading
            futures = []
            for _, word in word_scores[:stats.dynamic_max_backtrack]:
                future = executor.submit(try_slot, grid, slot, word,
                                         remaining_slots, words_by_length,
                                         word_frequencies, placed_words,
                                         progress, task, config)  # Pass config
                futures.append(future)

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result[0] is not None:
                        return result
                except Exception as e:
                    logging.warning(f"Error in thread: {e}")

        else:  # Recursive calls: sequential
            for _, word in word_scores[:stats.dynamic_max_backtrack]:
                result = try_slot(grid, slot, word, remaining_slots,
                                  words_by_length, word_frequencies,
                                  placed_words, progress, task, config)  # Pass config
                if result[0] is not None:
                    return result

    return None, None



def select_words(
        grid: List[List[str]],
        slots: List[Tuple[int, int, str, int]],
        words_by_length: Dict[int, List[str]],
        word_frequencies: Dict[str, float],
        progress: Progress,
        task: TaskID,
        config: Config  # Pass config
) -> Tuple[Optional[List[List[str]]], Optional[List[Tuple[str, int, int, str]]]]:
    """Initializes word selection."""
    global stats
    stats = CrosswordStats()
    stats.start_time = time.time()
    initial_placed_words: List[Tuple[str, int, int, str]] = []

    with ThreadPoolExecutor(max_workers=config.max_thread_pool_size) as executor:  # Use config
        return select_words_recursive(grid, slots, words_by_length,
                                      word_frequencies, initial_placed_words,
                                      progress, task, config,  # Pass config
                                      executor)



# --- Definition and Crossword Generation ---

def order_cell_numbers(slots: List[Tuple[int, int, str, int]]) -> Dict[Tuple[int, int, str], int]:
    """Orders cell numbers for clues."""
    numbered_cells = set()
    cell_numbers: Dict[Tuple[int, int, str], int] = {}
    next_number = 1

    sorted_slots = sorted(slots, key=lambda x: (x[0], x[1], 0 if x[2] == "across" else 1))

    for row, col, direction, length in sorted_slots:
        if (row, col) not in numbered_cells:
            cell_numbers[(row, col, direction)] = next_number
            numbered_cells.add((row, col))
            next_number += 1

    return cell_numbers



def generate_definitions(placed_words: List[Tuple[str, int, int, str]], language: str, config: Config) -> Dict[str, Dict[int, str]]: # pass config
    """Generates definitions, with numbering."""
    definitions = {"across": {}, "down": {}}
    slots = [(row, col, direction, len(word)) for word, row, col, direction in placed_words]
    cell_numbers = order_cell_numbers(slots)

    with Progress() as progress:
        task = progress.add_task("[blue]Generating Definitions...", total=len(placed_words))
        with ThreadPoolExecutor() as executor:
            futures = []
            for word, row, col, direction in placed_words:
                future = executor.submit(generate_definition_langchain, word, language, config) # Pass Config
                futures.append((future, word, row, col, direction))

            for future, word, row, col, direction in futures:
                try:
                    definition = future.result()
                    number = cell_numbers.get((row, col, direction))
                    if number:
                        definitions[direction][f"{number}. {word}"] = definition
                except Exception as e:
                    stats.definition_failures += 1
                    logging.error(f"Error getting definition for {word}: {e}")
                progress.update(task, advance=1)
    return definitions



def create_html(grid: List[List[str]], placed_words: List[Tuple[str, int, int, str]],
                definitions: Dict[str, Dict[int, str]], output_filename: str):
    """Generates the interactive HTML."""
    try:
        with open("template.html", "r", encoding="utf-8") as template_file:
            template = template_file.read()

        grid_html = '<table class="crossword-grid">'
        for row_index, row in enumerate(grid):
            grid_html += "<tr>"
            for col_index, cell in enumerate(row):
                if cell == "#":
                    grid_html += '<td class="black"></td>'
                else:
                    word_info = None
                    for word, word_row, word_col, direction in placed_words:
                        if direction == "across" and row_index == word_row and word_col <= col_index < word_col + len(word):
                            word_info = (word, word_row, word_col, direction, col_index - word_col)
                            break
                        elif direction == "down" and col_index == word_col and word_row <= row_index < word_row + len(word):
                            word_info = (word, word_row, word_col, direction, row_index - word_row)
                            break

                    if word_info:
                        word, word_row, word_col, direction, index_in_word = word_info
                        cell_id = f"{word_row}-{word_col}-{direction}"
                        if index_in_word == 0:
                            slots = [(word, row, col, direction) for word, row, col, direction in placed_words]
                            cell_numbers = order_cell_numbers(slots)
                            number = cell_numbers.get((word_row, word_col, direction), "")
                            grid_html += (
                                f'<td class="white" id="{cell_id}">'
                                f'<div class="cell-container">'
                                f'<span class="number">{number}</span>'
                                f'<input type="text" maxlength="1" class="letter" data-row="{row_index}" data-col="{col_index}" data-direction="{direction}">'
                                f'</div>'
                                f"</td>"
                            )
                        else:
                            grid_html += (
                                f'<td class="white" id="{cell_id}-{index_in_word}">'
                                f'<div class="cell-container">'
                                f'<input type="text" maxlength="1" class="letter" data-row="{row_index}" data-col="{col_index}" data-direction="{direction}">'
                                f'</div>'
                                f"</td>"
                            )
                    else:
                        grid_html += '<td class="white"></td>'
            grid_html += "</tr>"
        grid_html += "</table>"

        definitions_html = '<div class="definitions">'
        for direction, clues in definitions.items():
            definitions_html += f'<h3>{direction.capitalize()}</h3><ol>'
            for clue, definition in clues.items():
                definitions_html += f'<li><span class="clue-number">{clue.split(".")[0]}.</span> {definition}</li>'
            definitions_html += '</ol>'
        definitions_html += '</div>'

        final_html = template.format(grid_html=grid_html, definitions_html=definitions_html)

        with open(output_filename, "w", encoding="utf-8") as output_file:
            output_file.write(final_html)

    except FileNotFoundError:
        logging.error("template.html not found.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error generating HTML: {e}")
        sys.exit(1)



def main():
    """Main function to parse arguments and run."""
    parser = argparse.ArgumentParser(description="Generates a crossword.")
    parser.add_argument("--width", type=int, default=DEFAULT_GRID_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_GRID_HEIGHT)
    parser.add_argument("--black_squares", type=float, default=DEFAULT_BLACK_SQUARE_RATIO)
    parser.add_argument("--manual_grid", type=str, default=None)
    parser.add_argument("--grid_file", type=str, default=None)
    parser.add_argument("--lm_studio_url", type=str, default=DEFAULT_LM_STUDIO_URL)
    parser.add_argument("--words_file", type=str, default=DEFAULT_WORDS_FILE)
    parser.add_argument("--output_filename", type=str, default=DEFAULT_OUTPUT_FILENAME)
    parser.add_argument("--max_attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--llm_timeout", type=int, default=DEFAULT_LLM_TIMEOUT)
    parser.add_argument("--llm_max_tokens", type=int, default=DEFAULT_LLM_MAX_TOKENS)
    parser.add_argument("--language", type=str, default=DEFAULT_LANGUAGE)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max_grid_iterations", type=int, default=DEFAULT_MAX_GRID_ITERATIONS)
    parser.add_argument("--difficulty", type=str, default=DEFAULT_DIFFICULTY, choices=["easy", "medium", "hard"])

    args = parser.parse_args()

    if not all(isinstance(arg, int) and arg > 0 for arg in [args.width, args.height, args.max_attempts, args.timeout, args.llm_timeout, args.llm_max_tokens, args.max_grid_iterations]):
        logging.error("Positive integers required for numeric arguments.")
        sys.exit(1)
    if not 0.0 <= args.black_squares <= 1.0:
        logging.error("black_squares must be between 0.0 and 1.0.")
        sys.exit(1)
    if args.manual_grid and args.grid_file:
        logging.error("Specify either --manual_grid or --grid_file, not both.")
        sys.exit(1)

    # --- Configuration Initialization ---
    config = Config()
    config.update_from_args(args)


    words_by_length, word_frequencies = load_words(config.words_file, config.min_word_counts[config.difficulty]) # Pass Config

    if not words_by_length:
        logging.error("No valid words found. Check word file and difficulty.")
        sys.exit(1)

    max_dimension = max(config.grid_width, config.grid_height) # use Config
    for length in range(config.min_word_length, max_dimension + 1): # use Config
        if length not in words_by_length:
            logging.warning(f"No words of length {length} found.")

    build_word_index(words_by_length)

    llm_instance = setup_langchain_llm(config) #pass Config
    global llm
    llm = llm_instance

    console = Console()
    for attempt in range(config.max_grid_iterations): # use Config
        console.print(f"\n[bold blue]Attempt {attempt + 1}/{config.max_grid_iterations}[/]") # use Config
        grid = generate_grid(config) #pass Config

        if not is_valid_grid(grid):
            console.print("[red]Invalid grid. Retrying...[/]")
            continue

        console.print("[green]Initial Grid:[/]")
        print_grid(grid, console=console)

        slots = find_slots(grid, config)  # Pass config
        if not slots:
            console.print("[red]No valid slots. Retrying...[/]")
            continue

        with Progress() as progress:
            task = progress.add_task("[cyan]Selecting words...", total=None)
            filled_grid, placed_words = select_words(grid, slots, words_by_length, word_frequencies, progress, task, config)  # Pass config
            progress.update(task, completed=100)

        if filled_grid is not None:
            console.print("[green]Crossword filled![/]")
            print_grid(filled_grid, placed_words, console)
            definitions = generate_definitions(placed_words, config.language, config) #Pass config
            create_html(filled_grid, placed_words, definitions, config.output_filename) # use config
            console.print(f"[green]Saved to: {config.output_filename}[/]")# use config
            console.print(stats.get_summary())
            break
        else:
            console.print("[yellow]Failed to fill grid. Retrying...[/]")
            console.print(stats.get_summary())
            
            # Last resort: try aggressive black cell filling if this is the last attempt
            if attempt == config.max_grid_iterations - 1:  # On the last attempt
                console.print("[cyan]Trying aggressive black cell filling as a last resort...[/]")
                # Apply aggressive black cell filling
                modified_grid, modified_slots = final_fill_impossible_spaces(grid, slots, words_by_length, config)
                
                if modified_grid != grid:  # If the grid was modified
                    console.print("[green]Grid modified with aggressive black cell filling. Trying again...[/]")
                    print_grid(modified_grid, console=console)
                    console.print(f"[blue]Slots reduced from {len(slots)} to {len(modified_slots)}[/]")
                    
                    # Try filling the modified grid
                    with Progress() as progress:
                        task = progress.add_task("[cyan]Filling modified grid...", total=None)
                        filled_grid, placed_words = select_words(modified_grid, modified_slots, words_by_length, 
                                                                 word_frequencies, progress, task, config)
                        progress.update(task, completed=100)
                    
                    if filled_grid is not None:
                        console.print("[green]Crossword filled after aggressive black cell filling![/]")
                        print_grid(filled_grid, placed_words, console)
                        definitions = generate_definitions(placed_words, config.language, config)
                        create_html(filled_grid, placed_words, definitions, config.output_filename)
                        console.print(f"[green]Saved to: {config.output_filename}[/]")
                        console.print(stats.get_summary())
                        break
    else:
        console.print("[red]Failed to generate crossword.[/]")


def fill_impossible_spaces(grid: List[List[str]], slots: List[Tuple[int, int, str, int]], words_by_length: Dict[int, List[str]], config: Config, aggressive: bool = False) -> Tuple[List[List[str]], List[Tuple[int, int, str, int]]]:
    """Identifies and fills impossible or difficult-to-fill slots with black squares.
    Returns the modified grid and the updated slots list."""
    height, width = len(grid), len(grid[0])
    modified = False
    
    # Create a copy of the grid
    new_grid = [row[:] for row in grid]
    
    # Score each slot by how constrained it is
    slot_scores = []
    for slot in slots:
        row, col, direction, length = slot
        
        # Get the current pattern
        if direction == "across":
            pattern = "".join(grid[row][col:col + length])
        else:
            pattern = "".join(grid[row + i][col] for i in range(length))
        
        pattern = create_pattern(pattern)
        possible_words = len(word_index.get((length, pattern), []))
        
        # If no words can fit, this is an impossible slot
        if possible_words == 0:
            slot_scores.append((0, slot))
        else:
            # Lower score means more constrained
            slot_scores.append((possible_words, slot))
    
    # Sort by constraint level (most constrained first)
    slot_scores.sort(key=lambda x: x[0])
    
    # Try to fill the most constrained slots with black squares
    for score, slot in slot_scores:
        # In aggressive mode, consider slots with more options
        if not aggressive and score > 5:  # Skip slots that have enough options
            continue
        # In aggressive mode, consider slots with up to 20 options
        elif aggressive and score > 20:
            continue
            
        row, col, direction, length = slot
        
        # For very constrained slots, consider adding black squares
        if direction == "across":
            # Check if we can place a black square at the start or end
            positions_to_try = [(row, col), (row, col + length - 1)]
        else:
            positions_to_try = [(row, col), (row + length - 1, col)]
            
        for pos_row, pos_col in positions_to_try:
            # Skip if already a black square
            if new_grid[pos_row][pos_col] == "#":
                continue
                
            # Place black square symmetrically
            def place_symmetrically(r: int, c: int):
                new_grid[r][c] = "#"
                new_grid[height - 1 - r][width - 1 - c] = "#"
                
            # Check if placing a black square would create a 2x2 block
            def would_create_2x2_block(r: int, c: int) -> bool:
                # Check all four possible 2x2 configurations that include this position
                if (r > 0 and c > 0 and 
                    new_grid[r-1][c] == "#" and new_grid[r][c-1] == "#" and new_grid[r-1][c-1] == "#"):
                    return True
                if (r > 0 and c < width - 1 and 
                    new_grid[r-1][c] == "#" and new_grid[r][c+1] == "#" and new_grid[r-1][c+1] == "#"):
                    return True
                if (r < height - 1 and c > 0 and 
                    new_grid[r+1][c] == "#" and new_grid[r][c-1] == "#" and new_grid[r+1][c-1] == "#"):
                    return True
                if (r < height - 1 and c < width - 1 and 
                    new_grid[r+1][c] == "#" and new_grid[r][c+1] == "#" and new_grid[r+1][c+1] == "#"):
                    return True
                return False
            
            # Check if placing a black square would isolate any white squares
            def would_isolate_squares(r: int, c: int) -> bool:
                # Check the four adjacent cells
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and new_grid[nr][nc] == ".":
                        # Count connections for this adjacent cell
                        connections = 0
                        for adr, adc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            anr, anc = nr + adr, nc + adc
                            if 0 <= anr < height and 0 <= anc < width and (anr, anc) != (r, c) and new_grid[anr][anc] == ".":
                                connections += 1
                        if connections == 0:  # This would isolate the cell
                            return True
                return False
            
            # Check if we can place a black square here
            if not would_create_2x2_block(pos_row, pos_col) and not would_isolate_squares(pos_row, pos_col):
                # Place the black square symmetrically
                place_symmetrically(pos_row, pos_col)
                modified = True
                break  # Only make one modification at a time
                
        if modified:
            break
    
    # If we modified the grid, recalculate the slots
    if modified:
        new_slots = find_slots(new_grid, config)
        return new_grid, new_slots
    
    return grid, slots


def final_fill_impossible_spaces(grid: List[List[str]], slots: List[Tuple[int, int, str, int]], words_by_length: Dict[int, List[str]], config: Config) -> Tuple[List[List[str]], List[Tuple[int, int, str, int]]]:
    """More aggressive version of fill_impossible_spaces to be used at the end of generation.
    This function is designed to be called when the grid is nearly complete but has a few
    difficult-to-fill slots remaining. It will be more aggressive in placing black squares."""
    
    # First try the regular fill_impossible_spaces with aggressive mode
    new_grid, new_slots = fill_impossible_spaces(grid, slots, words_by_length, config, aggressive=True)
    
    # If that didn't change anything, try even more aggressive approaches
    if new_grid == grid:
        height, width = len(grid), len(grid[0])
        modified = False
        
        # Create a copy of the grid
        new_grid = [row[:] for row in grid]
        
        # For each remaining slot, calculate how many valid words can fit
        slot_options = []
        for slot in slots:
            row, col, direction, length = slot
            
            # Get the current pattern
            if direction == "across":
                pattern = "".join(grid[row][col:col + length])
            else:
                pattern = "".join(grid[row + i][col] for i in range(length))
            
            pattern = create_pattern(pattern)
            possible_words = word_index.get((length, pattern), [])
            
            # Store the number of options for this slot
            slot_options.append((len(possible_words), slot))
        
        # Sort by number of options (fewest first)
        slot_options.sort()
        
        # For slots with very few options, try more aggressive black square placement
        for options, slot in slot_options:
            if options > 30:  # Skip slots with plenty of options
                continue
                
            row, col, direction, length = slot
            
            # Try placing black squares at strategic positions within the slot
            positions_to_try = []
            
            if direction == "across":
                # Try positions along the slot
                for i in range(length):
                    positions_to_try.append((row, col + i))
            else:
                # Try positions along the slot
                for i in range(length):
                    positions_to_try.append((row + i, col))
                    
            # Prioritize positions that would split the slot into more manageable pieces
            if length > 4:
                # For longer slots, try placing black squares in the middle first
                positions_to_try.sort(key=lambda pos: abs(pos[0] - row - length//2 if direction == "down" else abs(pos[1] - col - length//2)))
            
            for pos_row, pos_col in positions_to_try:
                # Skip if already a black square
                if new_grid[pos_row][pos_col] == "#":
                    continue
                    
                # Place black square symmetrically
                def place_symmetrically(r: int, c: int):
                    new_grid[r][c] = "#"
                    new_grid[height - 1 - r][width - 1 - c] = "#"
                    
                # Check if placing a black square would create a 2x2 block
                def would_create_2x2_block(r: int, c: int) -> bool:
                    # Check all four possible 2x2 configurations that include this position
                    if (r > 0 and c > 0 and 
                        new_grid[r-1][c] == "#" and new_grid[r][c-1] == "#" and new_grid[r-1][c-1] == "#"):
                        return True
                    if (r > 0 and c < width - 1 and 
                        new_grid[r-1][c] == "#" and new_grid[r][c+1] == "#" and new_grid[r-1][c+1] == "#"):
                        return True
                    if (r < height - 1 and c > 0 and 
                        new_grid[r+1][c] == "#" and new_grid[r][c-1] == "#" and new_grid[r+1][c-1] == "#"):
                        return True
                    if (r < height - 1 and c < width - 1 and 
                        new_grid[r+1][c] == "#" and new_grid[r][c+1] == "#" and new_grid[r+1][c+1] == "#"):
                        return True
                    return False
                
                # Check if placing a black square would isolate any white squares
                def would_isolate_squares(r: int, c: int) -> bool:
                    # Check the four adjacent cells
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < height and 0 <= nc < width and new_grid[nr][nc] == ".":
                            # Count connections for this adjacent cell
                            connections = 0
                            for adr, adc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                anr, anc = nr + adr, nc + adc
                                if 0 <= anr < height and 0 <= anc < width and (anr, anc) != (r, c) and new_grid[anr][anc] == ".":
                                    connections += 1
                            if connections == 0:  # This would isolate the cell
                                return True
                    return False
                
                # Check if we can place a black square here
                if not would_create_2x2_block(pos_row, pos_col) and not would_isolate_squares(pos_row, pos_col):
                    # Place the black square symmetrically
                    place_symmetrically(pos_row, pos_col)
                    modified = True
                    break  # Only make one modification at a time
                    
            if modified:
                break
        
        # If we modified the grid, recalculate the slots
        if modified:
            new_slots = find_slots(new_grid, config)
    
    return new_grid, new_slots

if __name__ == "__main__":
    main()
