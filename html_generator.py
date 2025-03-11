from typing import List, Tuple, Dict

def create_html(
    grid: List[List[str]],
    placed_words: List[Tuple[str, int, int, str]],
    definitions: Dict[str, Dict[int, str]],
    filename: str,
):
    """Generates the HTML for the interactive crossword."""
    with open(filename, "w", encoding="utf-8") as f:
        _write_html_header(f)
        _write_html_body(f, grid, placed_words, definitions)
        _write_html_footer(f)

def _write_html_header(f) -> None:
    """Writes HTML head section with styles and scripts."""
    f.write("<!DOCTYPE html>\n")
    f.write("<html lang='it'>\n")
    f.write("<head>\n")
    f.write("  <meta charset='UTF-8'>\n")
    f.write("  <meta name='viewport' content='width=device-width, initial-scale=1.0'>\n")
    f.write("  <title>Cruciverba</title>\n")
    _write_styles(f)
    _write_scripts(f)
    f.write("</head>\n")

def _write_styles(f) -> None:
    """Writes CSS styles."""
    f.write("  <style>\n")
    f.write("    body { font-family: sans-serif; display: flex; flex-direction: column; align-items: center; margin: 20px; }\n")
    f.write("    h1 { text-align: center; }\n")
    f.write("    .crossword-container { display: flex; flex-direction: column; align-items: center; width: 100%; max-width: 600px; }\n")
    f.write("    table { border-collapse: collapse; margin-bottom: 20px; }\n")
    f.write("    td { border: 1px solid black; width: 30px; height: 30px; text-align: center; font-size: 20px; position: relative; }\n")
    f.write("    .black { background-color: black; }\n")
    f.write("    input { box-sizing: border-box; border: none; width: 100%; height: 100%; text-align: center; font-size: 20px; padding: 0; outline: none; background-color: transparent; }\n")
    f.write("    input:focus { background-color: #e0e0e0; }\n")
    f.write("    .number { position: absolute; top: 2px; left: 2px; font-size: 8px; }\n")
    f.write("    .definitions-container { width: 100%; max-width: 600px; }\n")
    f.write("    .definitions { margin-top: 1em; }\n")
    f.write("    h2, h3 { text-align: center; }\n")
    f.write("    ul { list-style-type: none; padding: 0; }\n")
    f.write("    li { margin-bottom: 0.5em; }\n")
    f.write("  </style>\n")

def _write_scripts(f) -> None:
    """Writes JavaScript functions."""
    f.write("  <script>\n")
    f.write("    function navigateCells(event) {\n")
    f.write("      const input = event.target;\n")
    f.write(
        "      const maxLength = parseInt(input.getAttribute('maxlength'), 10);\n"
    )
    f.write("      if (input.value.length >= maxLength) {\n")
    f.write("        const currentRow = input.parentElement.parentElement;\n")
    f.write(
        "        const currentCellIndex = Array.from(currentRow.children).indexOf(input.parentElement);\n"
    )
    f.write("        const nextCell = currentRow.children[currentCellIndex + 1];\n")
    f.write("        if (nextCell) {\n")
    f.write("          const nextInput = nextCell.querySelector('input');\n")
    f.write("          if (nextInput) {\n")
    f.write("            nextInput.focus();\n")
    f.write("          }\n")
    f.write("        } else {\n")
    f.write("          const nextRow = currentRow.nextElementSibling;\n")
    f.write("          if (nextRow) {\n")
    f.write(
        "          const nextRowFirstCell = nextRow.children[currentCellIndex];\n"
    )
    f.write("              if (nextRowFirstCell) {\n")
    f.write(
        "                  const nextInput = nextRowFirstCell.querySelector('input');\n"
    )
    f.write("                  if(nextInput) { nextInput.focus(); } \n")
    f.write("              }\n")
    f.write("          }\n")
    f.write("        }\n")
    f.write("      }\n")
    # Handle arrow keys
    f.write("      switch (event.key) {\n")
    f.write("        case 'ArrowUp':\n")
    f.write("        case 'Up':\n")
    f.write("          moveFocus(-1, 0, input);\n")
    f.write("          break;\n")
    f.write("        case 'ArrowDown':\n")
    f.write("        case 'Down':\n")
    f.write("          moveFocus(1, 0, input);\n")
    f.write("          break;\n")
    f.write("        case 'ArrowLeft':\n")
    f.write("        case 'Left':\n")
    f.write("          moveFocus(0, -1, input);\n")
    f.write("          break;\n")
    f.write("        case 'ArrowRight':\n")
    f.write("        case 'Right':\n")
    f.write("          moveFocus(0, 1, input);\n")
    f.write("          break;\n")
    f.write("      }\n")
    f.write("    }\n")

    f.write("    function moveFocus(rowDelta, colDelta, currentInput) {\n")
    f.write("        const currentRow = currentInput.parentElement.parentElement;\n")
    f.write(
        "        const currentCellIndex = Array.from(currentRow.children).indexOf(currentInput.parentElement);\n"
    )
    f.write(
        "        const allRows = Array.from(currentRow.parentElement.children);\n"
    )
    f.write("        const currentRowIndex = allRows.indexOf(currentRow);\n")
    f.write("        const nextRowIndex = currentRowIndex + rowDelta;\n")
    f.write("        if (nextRowIndex >= 0 && nextRowIndex < allRows.length) {\n")
    f.write("            const nextRow = allRows[nextRowIndex];\n")
    f.write("            const nextCellIndex = currentCellIndex + colDelta;\n")
    f.write(
        "            if (nextCellIndex >= 0 && nextCellIndex < nextRow.children.length) {\n"
    )
    f.write("                const nextCell = nextRow.children[nextCellIndex];\n")
    f.write("                const nextInput = nextCell.querySelector('input');\n")
    f.write("                if (nextInput) {\n")
    f.write("                    nextInput.focus();\n")
    f.write("                }\n")
    f.write("            }\n")
    f.write("        }\n")
    f.write("    }\n")
    f.write("  </script>\n")

def _write_html_body(
    f,
    grid: List[List[str]], 
    placed_words: List[Tuple[str, int, int, str]],
    definitions: Dict[str, Dict[int, str]]
) -> None:
    """Writes HTML body with crossword grid and definitions."""
    f.write("<body>\n")
    f.write("  <h1>Cruciverba</h1>\n")
    
    # Write crossword grid
    _write_crossword_grid(f, grid, _create_cell_numbers(placed_words))
    
    # Write definitions
    _write_definitions(f, definitions)
    
    f.write("</body>\n")

def _create_cell_numbers(
    placed_words: List[Tuple[str, int, int, str]]
) -> Dict[Tuple[int, int], int]:
    """Creates mapping of cell positions to numbers."""
    cell_numbers: Dict[Tuple[int, int], int] = {}
    next_number = 1
    for word, row, col, direction in placed_words:
        if (row, col) not in cell_numbers:
            cell_numbers[(row, col)] = next_number
            next_number += 1
    return cell_numbers

def _write_crossword_grid(
    f,
    grid: List[List[str]],
    cell_numbers: Dict[Tuple[int, int], int]
) -> None:
    """Writes the crossword grid table."""
    f.write("  <div class='crossword-container'>\n")
    f.write("    <table>\n")
    
    for r, row in enumerate(grid):
        f.write("  <tr>\n")
        for c, cell in enumerate(row):
            if cell == "#":
                f.write("    <td class='black'></td>\n")
            else:
                cell_number_html = ""
                for (word_row, word_col), num in cell_numbers.items():
                    if word_row == r and word_col == c:
                        cell_number_html = f"<span class='number'>{num}</span>"
                        break
                f.write(f"    <td>{cell_number_html}<input type='text' maxlength='1' oninput='this.value=this.value.toUpperCase(); navigateCells(event);'></td>\n")
        f.write("  </tr>\n")
    f.write("    </table>\n")
    f.write("  </div>\n")

def _write_definitions(
    f,
    definitions: Dict[str, Dict[int, str]]
) -> None:
    """Writes the definitions sections."""
    f.write("  <div class='definitions-container'>\n")
    f.write("     <h2>Definizioni</h2>\n")

    # Write across definitions
    f.write("     <div class='definitions'>\n")
    f.write("     <h3>Orizzontali</h3>\n")
    f.write("     <ul>\n")
    for key, definition in definitions["across"].items():
        f.write(f"       <li><b>{key}</b> {definition}</li>\n")
    f.write("     </ul>\n")
    f.write("     </div>\n")

    # Write down definitions
    f.write("     <div class='definitions'>\n")
    f.write("     <h3>Verticali</h3>\n")
    f.write("     <ul>\n")
    for key, definition in definitions["down"].items():
        f.write(f"       <li><b>{key}</b> {definition}</li>\n")
    f.write("     </ul>\n")
    f.write("     </div>\n")
    f.write("  </div>\n")

def _write_html_footer(f) -> None:
    """Writes HTML closing tags."""
    f.write("</html>\n")