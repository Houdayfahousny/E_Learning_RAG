# Elearning Platform LLM Avatar

This project is an interactive e-learning content generator that leverages Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs) to create educational slides enriched with visual content (images and diagrams). It supports multiple domains (e.g., Java, Angular, Spring/JEE) and languages (French and English).

## Features

- **RAG-based content generation**: Combines LLMs with domain-specific knowledge for accurate, contextual slide creation.
- **Automatic image generation**: Integrates with Stable Diffusion (local or Stability AI) and DALL·E APIs to create technical diagrams and illustrations.
- **Mermaid diagram creation**: Generates Mermaid.js diagrams for visual explanations of technical topics.
- **Multi-language support**: Generates content in French or English.
- **Domain extensibility**: Easily add new domains or training content via JSON files.
- **JSON export**: Saves generated slides, images, and diagrams for further use.

## Project Structure

```
Model_Training/
  enhanced_llama3_model.py   # Main script for enhanced RAG + visual content
  build_faiss_index.py       # FAISS index builder for semantic search
  Llama3_model.py            # Base Llama3 model logic
faiss_index/
  *.index                    # FAISS vector indices for each domain
RAG_Content/
  *.json                     # Training content for each domain
docs/
  *.json                     # Documentation and slide content
Summary Output/
  *.json                     # Generated summaries
Explanation Output/
  *.json                     # Generated explanations
```

## Requirements

- Python 3.8+
- [sentence-transformers](https://www.sbert.net/)
- [faiss](https://github.com/facebookresearch/faiss)
- [requests](https://docs.python-requests.org/)
- [numpy](https://numpy.org/)
- Access to Stable Diffusion (local Automatic1111 or Stability AI API) and/or OpenAI DALL·E API

Install dependencies:
```sh
pip install -r requirements.txt
```

## Usage

1. **Configure API Keys**  
   Edit `enhanced_llama3_model.py` and set your API keys in the `IMAGE_APIS` dictionary.

2. **Prepare Content**  
   Add or update training content in `RAG_Content/` and documentation in `docs/`.

3. **Build FAISS Indexes**  
   Run:
   ```sh
   python Model_Training/build_faiss_index.py
   ```

4. **Generate Slides**  
   Run the main script:
   ```sh
   python Model_Training/enhanced_llama3_model.py
   ```
   Follow the prompts to select language, subject, and level.

5. **Output**  
   - Generated images: `./images/`
   - Mermaid diagrams: `./diagrams/`
   - Enriched slides (JSON): output file as specified

## Customization

- **Add new domains**: Place new training JSON files in `RAG_Content/` and update `AVAILABLE_DOMAINS` in the script.
- **Change prompts**: Edit `ENHANCED_SYSTEM_PROMPT` and `IMAGE_DESCRIPTION_PROMPTS` in `enhanced_llama3_model.py`.

## License

This project is for educational and research purposes.

---

*For more details, see the code