# PHPAiModel-Transformer


PHPAiModel-Transformer is a lightweight, pure PHP implementation of a character-level Transformer model runtime. It includes a simple chat UI for interacting with models, supporting inference on JSON-encoded weights. Designed for PHP 7.4+, it features pre-LayerNorm, GELU activation, Multi-Head Attention (MHA), and a tied output head for efficient generation.

This project enables running Transformer-based language models directly in PHP without external dependencies, ideal for educational purposes, prototyping, or lightweight deployments.

## Overview
PHPAiModel-Transformer provides a backend runtime (`aicore.php`) for Transformer inference and a frontend chat interface (`index.php`). The runtime loads model weights from JSON files, performs forward passes for logit computation, and generates text using temperature and top-k sampling. The chat UI allows selecting models, setting parameters, and interacting via prompts.

Key components:
- **Character-level Tokenization**: Simple mapping of characters to IDs.
- **Causal Self-Attention**: Ensures autoregressive generation.
- **Feed-Forward Layers**: With GELU activation and configurable dimensions.
- **Generation**: Supports context management up to `max_seq` length.

## Features
- Pure PHP implementation (no extensions required beyond standard ones like `mbstring` and `json`).
- Supports pre-LayerNorm architecture with Multi-Head Attention.
- Tied embeddings for input and output (efficient memory use).
- Configurable sampling: temperature and top-k.
- Simple web-based chat UI with model selection and parameter controls.
- Debug metadata in responses (e.g., model dimensions).
- MIT licensed, open-source.

## Requirements
- PHP 7.4 or higher (tested for compatibility).
- Web server (e.g., Apache, Nginx) for serving the UI.
- Recommended: Increase `memory_limit` to 2048M or more for larger models.
- Recommended: Set `max_execution_time` to 600 seconds for long generations.

![PHPAiModel-Transformer](screen.png)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.