import json
import tempfile
import unittest
from pathlib import Path

from sharegpt_converter import (
    ConversationEntry,
    build_sharegpt_entries,
    convert_to_sharegpt,
    get_conversation_thread,
    normalize_content,
)


def build_mapping(messages):
    mapping = {}
    previous_id = None
    for index, (role, text) in enumerate(messages, start=1):
        node_id = str(index)
        mapping[node_id] = {
            "id": node_id,
            "message": {
                "author": {"role": role},
                "content": {"parts": [text]},
            },
            "parent": previous_id,
            "children": [],
        }
        if previous_id:
            mapping[previous_id]["children"].append(node_id)
        previous_id = node_id
    return mapping, previous_id


class ShareGPTConverterTests(unittest.TestCase):
    def test_normalize_content_strips_empty_entries(self):
        result = normalize_content(["Hello", "", None, "World"])  # type: ignore[list-item]
        self.assertEqual(result, "Hello\nWorld")

    def test_reconstructs_thread_from_mapping(self):
        mapping, current_node = build_mapping([("user", "Hi"), ("assistant", "Hello!")])
        thread = get_conversation_thread({"mapping": mapping, "current_node": current_node})
        self.assertEqual(len(thread), 2)
        self.assertIsInstance(thread[0], ConversationEntry)
        self.assertEqual(thread[0].role, "user")
        self.assertEqual(thread[1].content, "Hello!")

    def test_converts_file_to_sharegpt_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            mapping, current_node = build_mapping(
                [("user", "What is the weather?"), ("assistant", "Sunny today.")]
            )
            conversation = {"mapping": mapping, "current_node": current_node}
            (base / "2023-test.json").write_text(json.dumps(conversation), encoding="utf-8")

            output_path = base / "out.jsonl"
            written = convert_to_sharegpt(str(base / "2023-*.json"), str(output_path), min_turns=2)

            self.assertEqual(written, 1)
            content = output_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(content), 1)
            payload = json.loads(content[0])
            self.assertEqual(len(payload["conversations"]), 2)
            self.assertEqual(payload["conversations"][0]["from"], "human")
            self.assertEqual(payload["conversations"][1]["from"], "gpt")
            self.assertTrue(payload["source"].endswith("2023-test.json"))

    def test_skips_conversations_below_min_turns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            mapping, current_node = build_mapping([("user", "Solo message")])
            (base / "2023-short.json").write_text(
                json.dumps({"mapping": mapping, "current_node": current_node}), encoding="utf-8"
            )
            output_path = base / "out.jsonl"
            written = convert_to_sharegpt(str(base / "2023-*.json"), str(output_path), min_turns=2)

            self.assertEqual(written, 0)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "")

    def test_build_sharegpt_entries_structure(self):
        entries = [ConversationEntry("user", "Hello"), ConversationEntry("assistant", "Hi there")]
        payload = build_sharegpt_entries(entries, Path("example.json"))
        self.assertEqual(payload["conversations"][0]["from"], "human")
        self.assertEqual(payload["conversations"][1]["from"], "gpt")
        self.assertEqual(payload["source"], "example.json")


if __name__ == "__main__":
    unittest.main()
