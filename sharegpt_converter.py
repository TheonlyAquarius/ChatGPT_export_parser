"""Convert ChatGPT export JSON files into ShareGPT-style JSONL.

The module exposes a small CLI so you can point it at the files you
exported from ChatGPT (e.g., ``2023-*.json``) and generate a single
``.jsonl`` file suitable for training pipelines that expect the
ShareGPT schema.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

# Default configuration mirrors the values used by the original script.
DEFAULT_INPUT_PATTERN = "2023-*.json"
DEFAULT_OUTPUT_FILE = "training_data_sharegpt.jsonl"
DEFAULT_MIN_TURNS = 2

logger = logging.getLogger(__name__)


class ConversationEntry:
    """Structured representation of a single message turn."""

    __slots__ = ("role", "content")

    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content

    def to_sharegpt(self) -> Dict[str, str]:
        return {
            "from": "human" if self.role == "user" else "gpt",
            "value": self.content,
        }


def normalize_content(parts: Sequence[object]) -> str:
    """Return a clean text string from a sequence of content parts."""

    safe_parts: List[str] = []
    for part in parts or []:
        if isinstance(part, str) and part.strip():
            safe_parts.append(part)
    # Joining with newlines keeps distinct parts legible if the export
    # included multiple blocks of text.
    return "\n".join(safe_parts)


def get_conversation_thread(conversation: Dict) -> List[ConversationEntry]:
    """Reconstruct the active conversation branch from a ChatGPT export.

    The export format contains a ``mapping`` of nodes and a ``current_node``
    that points to the leaf of the active branch. We walk backwards via the
    ``parent`` references until we reach the root, collecting user/assistant
    messages along the way.
    """

    current_node_id = conversation.get("current_node")
    mapping: Dict[str, Dict] = conversation.get("mapping") or {}

    if not current_node_id or not mapping:
        return []

    thread: List[ConversationEntry] = []
    visited: set[str] = set()

    while current_node_id and current_node_id not in visited:
        visited.add(current_node_id)
        node = mapping.get(current_node_id)
        if not node:
            break

        message = node.get("message") or {}
        role = (message.get("author") or {}).get("role")
        content = normalize_content((message.get("content") or {}).get("parts") or [])

        if content and role in {"user", "assistant"}:
            thread.append(ConversationEntry(role=role, content=content))

        current_node_id = node.get("parent")

    return list(reversed(thread))


def iterate_conversations(data: object) -> Iterable[Dict]:
    """Yield raw conversation dictionaries from the loaded JSON payload."""

    if isinstance(data, dict) and data.get("mapping"):
        yield data
    elif isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict) and entry.get("mapping"):
                yield entry


def build_sharegpt_entries(conversations: Iterable[ConversationEntry], source: Path) -> Dict:
    """Return the ShareGPT-formatted dictionary for a single conversation."""

    return {
        "conversations": [entry.to_sharegpt() for entry in conversations],
        "source": str(source),
    }


def convert_to_sharegpt(input_pattern: str, output_file: str, min_turns: int) -> int:
    """Convert matching ChatGPT export files into a ShareGPT-style JSONL.

    Returns the number of conversations written to ``output_file``.
    """

    pattern_path = Path(input_pattern)
    if pattern_path.is_absolute():
        input_paths = sorted(pattern_path.parent.glob(pattern_path.name))
    else:
        input_paths = sorted(Path().glob(input_pattern))
    if not input_paths:
        logger.warning("No files matched the pattern %s", input_pattern)
        return 0

    written = 0
    with open(output_file, "w", encoding="utf-8") as outfile:
        for path in input_paths:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                logger.error("Skipping %s (invalid JSON): %s", path, exc)
                continue

            for conversation in iterate_conversations(data):
                thread = get_conversation_thread(conversation)
                if len(thread) < min_turns:
                    continue
                entry = build_sharegpt_entries(thread, path)
                outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")
                written += 1

    logger.info("Wrote %s conversations to %s", written, output_file)
    return written


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ChatGPT export files into ShareGPT-compatible JSONL.",
    )
    parser.add_argument(
        "--input-pattern",
        default=DEFAULT_INPUT_PATTERN,
        help="Glob pattern for input JSON files (default: %(default)s)",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help="Destination JSONL file (default: %(default)s)",
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=DEFAULT_MIN_TURNS,
        help="Minimum number of user/assistant turns required to emit a conversation (default: %(default)s)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)
    convert_to_sharegpt(
        input_pattern=args.input_pattern,
        output_file=args.output_file,
        min_turns=args.min_turns,
    )


if __name__ == "__main__":
    main()
