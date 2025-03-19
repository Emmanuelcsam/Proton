# Deep Research Assistant

A comprehensive research tool that combines advanced search capabilities, deep web crawling, intelligent content extraction, and multi-model summarization to produce detailed research reports on any topic.

## Features

- **Advanced Search**: Uses DuckDuckGo to find relevant information about your research topic
- **Deep Web Crawling**: Leverages Photon to discover related content beyond initial search results
- **Intelligent Content Extraction**: Extracts and processes text from various web sources
- **Multi-Model Summarization**: Offers multiple summarization methods:
  - TextRank (built-in algorithm, no API required)
  - Ollama (local AI models like gemma3:12b, llama3:latest, deepseek-r1:8b)
  - OpenAI (using your API key)
- **Source Hierarchy Tracking**: Tracks and visualizes how sources were discovered
- **Comprehensive Research Reports**: Generates detailed, well-structured reports with proper citations

## Installation

### Prerequisites

- Python 3.7 or higher
- Photon (included or as a dependency)
- [Optional] Ollama for local AI model summarization
- [Optional] OpenAI API key for GPT-based summarization

### Basic Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/deep-research-assistant.git
   cd deep-research-assistant
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install NLTK data (for improved text processing):
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
   ```

4. [Optional] Install and set up Ollama for local AI summarization:
   ```bash
   # Install Ollama from https://ollama.ai/
   # Then pull required models:
   ollama pull gemma3:12b
   ollama pull llama3:latest
   ollama pull deepseek-r1:8b
   ```

## Usage

### Basic Usage

Run a research query with default settings:

```bash
python deep_research_assistant.py "quantum computing applications"
```

### Advanced Usage

Customize your research with various options:

```bash
python deep_research_assistant.py --depth 4 --method ollama --model gemma3:12b --summary-depth detailed "artificial intelligence ethics"
```

### Options

- `--depth`: Research depth (1-5), where 1 is a quick overview and 5 is a comprehensive deep dive
- `--method`: Summarization method (`textrank`, `ollama`, `openai`)
- `--model`: Model to use with Ollama or OpenAI
- `--summary-depth`: Level of detail in summaries (`short`, `medium`, `detailed`)
- `--format`: Output format (`markdown`, `text`, `json`)
- `--output-dir`: Directory for output files
- `--max-sources`: Maximum number of sources to analyze (0 for auto-scaling based on depth)
- `--exclude-domains`: Domains to exclude from search/crawl results
- `--ollama-url`: URL for Ollama API
- `--openai-key`: OpenAI API key
- `--photon-path`: Path to photon.py script
- `--verbose`: Enable verbose output
- `--debug`: Enable debug mode

## Output Example

The tool generates comprehensive research reports that include:

- Executive summary of the topic
- Key findings from top sources
- Detailed source summaries with proper citations
- Source hierarchy visualization
- Research methodology

### Example Markdown Output

```markdown
# Research Report: Quantum Computing Applications

*Generated on: 2025-03-17 15:30:45*

## Research Metadata

- **Query:** Quantum Computing Applications
- **Sources:** 23
- **Source Types:** 8 articles, 6 research_papers, 5 blog_posts, 4 documentation
- **Average Source Quality:** 78.4/100
- **Average Source Credibility:** 82.1/100

## Executive Summary

Quantum computing applications span multiple industries with significant potential for transformative impact. Finance applications include portfolio optimization, risk analysis, and fraud detection that can process complex scenarios exponentially faster than classical computers. In healthcare, quantum computing enables faster drug discovery through molecular modeling and more accurate genomic analysis. Machine learning benefits from quantum algorithms like quantum neural networks and quantum support vector machines, potentially solving problems currently intractable for classical systems. Materials science applications involve simulating quantum systems to develop new materials with desirable properties. However, most quantum computing applications remain theoretical or early-stage due to hardware limitations and error rates in current quantum processors [S3, S7, S12]. Experts project commercially viable quantum advantage in specific domains within 3-5 years, with more general applications emerging over the next decade [S1].

...
```

## How It Works

1. **Search**: Finds initial information sources using DuckDuckGo
2. **Crawl**: Discovers additional related sources through Photon web crawling
3. **Extract**: Pulls content from discovered sources using intelligent content extraction
4. **Analyze**: Evaluates sources for relevance, quality, and credibility
5. **Summarize**: Generates concise summaries of each source using the selected method
6. **Synthesize**: Creates a comprehensive research report with proper citations

## License

MIT

## Acknowledgments

- Photon: https://github.com/s0md3v/Photon
- DDGR: https://github.com/jarun/ddgr
- TextRank: Inspired by the research paper "TextRank: Bringing Order into Texts" by Mihalcea and Tarau
