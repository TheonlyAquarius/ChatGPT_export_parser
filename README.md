# ChatHistoryToProject - Add all your chat history to ChatGPT's Project

**ChatHistoryToProject** is a streamlined tool designed to simplify the process of preparing your OpenAI chat history for use with ChatGPT’s Projects feature. By stripping unnecessary details and splitting data into smaller, manageable files, ChatHistoryToProject ensures your chat history is ready for seamless integration.

---

## ✨ Features

- **Efficient Data Cleaning**: Removes metadata and extraneous fields, keeping only essential conversation content.
- **Automated Splitting**: Divides large JSON files into up to 20 smaller files to comply with ChatGPT’s file upload limits.
- **Plug-and-Play Simplicity**: Operates directly within the folder containing your exported chat data, no additional configuration required.

---

## 🛠 Prerequisites

1. **Python**: Ensure Python 3.7 or higher is installed on your system.

---

## 🚀 How to Use

### 1. Export Your OpenAI Chat History
   - Navigate to your ChatGPT account settings and export your chat data. (Settings -> Data Controls -> Export Data)
   - Extract the downloaded ZIP file into a folder, ensuring the `conversations.json` file is present.

### 2. Set Up the Script
   - Download the `ChatHistoryToProject` script.
   - Place the script in the same folder as `conversations.json`.

### 3. Run the Script
   - Open a terminal or command prompt.
   - Navigate to the folder containing the script and `conversations.json`.
   - Execute the following command:
     ```
     python ChatHistoryToProject.py
     ```

### 4. Review the Output
   - The script creates an `Output` folder in the same directory.
   - Up to 20 JSON files (`part_1.json`, `part_2.json`, etc.) will be generated, ready to upload to your ChatGPT Project.


---

## 🤔 Why Use ChatHistoryToProject?

The new ChatGPT Projects feature allows you to upload up to 20 files for advanced analysis and interaction. If you upload all your chat history to the Project and then start to chat with it, it will enchance your working experience as ChatGPT will know **everything** you discussed

---

## 🌍 Contributions Welcome!

I welcome contributions to improve ChatHistoryToProject. If you have ideas, encounter issues, or want to enhance the tool, feel free to open a pull request or submit an issue on the GitHub repository. Collaboration helps make this tool better for everyone!

---

Happy Chatting with Projects! 🚀


---

## 📚 Export to ShareGPT JSONL

If you want to turn your exported chats into ShareGPT-style training data, use the `sharegpt_converter.py` helper.

1. Place `sharegpt_converter.py` alongside your exported `2023-*.json` files (or adjust the glob to match your filenames).
2. Run the converter:

   ```bash
   python sharegpt_converter.py --input-pattern "2023-*.json" --output-file training_data_sharegpt.jsonl
   ```

   Use `--verbose` for extra logging or `--min-turns` to require more user/assistant turns before a conversation is emitted.

The script writes one ShareGPT-formatted conversation per line to the specified JSONL file.
