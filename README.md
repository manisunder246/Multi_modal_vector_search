# Multi-Modal Vector Search from PDF Files

This Jupyter notebook implements a **multi-modal vector search** system to perform information retrieval from PDF documents. It uses **vector embeddings** to search for tabular, text and image data within PDFs, enabling enhanced retrieval accuracy and relevance using LangChain,and GPT-40 . The notebook showcases the step-by-step process to achieve this functionality, integrating multiple tools, libraries, and techniques to extract, transform, and visualize information.

## Table of Contents

1. [Overview](#overview)
2. [Key Libraries and Dependencies](#key-libraries-and-dependencies)
3. [Setup and Initialization](#setup-and-initialization)
4. [PDF Data Extraction](#pdf-data-extraction)
5. [Text Preprocessing](#text-preprocessing)
6. [Image Preprocessing](#image-preprocessing)
7. [Embedding Generation](#embedding-generation)
8. [Vector Database Search](#vector-database-search)
9. [Query Processing](#query-processing)
10. [Creating the RAG Pipeline](#creating-the-rag-pipeline)
11. [Conclusion](#conclusion)

## Overview

This notebook demonstrates how to extract data from PDF files, convert the text and images into embeddings, store them in a vector database, and conduct searches based on natural language queries. It is designed to handle **multi-modal data**, which includes both textual and image content, ensuring that relevant information is retrieved, regardless of its format within the document.

## Key Libraries and Dependencies

The following libraries and tools are used:

- **langchain**: For building LLM-based applications and handling embeddings.
- **faiss**: For efficient similarity search and clustering of dense vectors.
- **pdfplumber** and **fitz**: For extracting text and images from PDF files.
- **pytesseract**: For Optical Character Recognition (OCR) on images within PDFs.
- **openai**: For generating text embeddings using OpenAI's models.
- **unstructured**: For text parsing and formatting.
- **gpt4all**: For integrating GPT-based models in the search pipeline.
- **langchain_community** and **langchain_openai**: For advanced integrations and model management.
- **function_calling1**: For custom parsing of Google-style docstrings using `_parse_google_docstring`.

## Setup and Initialization

The notebook begins with importing necessary libraries and installing missing dependencies via `pip`. The environment is set up to ensure that all tools and packages are available for subsequent operations.

- The notebook installs required libraries such as `langchain`, `faiss`, `pytesseract`, and others.
- Environment variables are set up for API keys and other configurations needed to access external services like OpenAI models.

## PDF Data Extraction

This section focuses on extracting data from the PDF files:

- **Text Extraction**: 
  - Uses libraries like `pdfplumber` and `fitz` to extract text data from each page of the PDF.
  - The extracted text is cleaned and stored for further processing.
  
- **Image Extraction**: 
  - Extracts images embedded within the PDF pages using `fitz`.
  - The images are saved and prepared for OCR and vectorization.

## Text Preprocessing

Once the text is extracted, it undergoes preprocessing:

- **Tokenization**: The text is split into tokens to standardize the format.
- **Cleaning**: Removes special characters, unnecessary spaces, and formatting issues.
- **Normalization**: Converts the text to lowercase for consistency.

## Image Preprocessing

The images extracted from the PDF are also processed:

- **OCR (Optical Character Recognition)**: 
  - Uses `pytesseract` to convert images to text.
  - Extracted text from images is then added to the text corpus for vectorization.

- **GPT-4 Integration**: 
  - Generates detailed summaries of images based on OCR text, image content, and pre-defined prompts.
  - Uses OpenAI's GPT-4 model to provide a 5-point summary of each image, incorporating graph analysis, labels, and key metrics.

- **Image Embeddings**: 
  - Converts the processed images into embeddings using OpenAI embeddings instead of transformers.

## Embedding Generation

The text and image data are transformed into embeddings:

- **Text and Image Embeddings**: 
  - Uses **OpenAI's embeddings** to convert text and image data into vector embeddings. 
  - The embeddings are used for multi-modal vector search, replacing the previous reliance on **transformers**.

## Vector Database Search

The generated embeddings are stored in a vector database (e.g., FAISS):

- **Indexing**: Embeddings are indexed for efficient similarity search.
- **Storage**: The vector database is built using FAISS, allowing for fast and scalable retrieval.

## Query Processing

The user can input a natural language query, which is processed to retrieve relevant data:

- The query is converted into an embedding using OpenAI's embedding model.
- A similarity search is performed in the vector database to find the most relevant embeddings.
- The search results include text snippets, image references, and other relevant data.

## 10. Creating the RAG Pipeline

This section sets up a **Retrieval-Augmented Generation (RAG) pipeline**, which uses retrieved information from the vector store to generate specific answers to user queries using GPT-4. The pipeline integrates multiple components, including a prompt, retriever, and language model, to handle natural language queries and return accurate responses.

- **Custom Parsing**: Uses a custom module (`function_calling1`) to parse Google-style docstrings, as the native LangChain library did not support this feature.
- **OpenAI Embeddings for Retrieval**: Uses OpenAI embeddings to retrieve relevant context from the vector store.
- **Chaining Components**: The retriever, prompt, and language model are chained together to generate accurate answers based on the retrieved context.

## Conclusion

This notebook demonstrates a comprehensive approach to building a **multi-modal vector search** system from PDF files. It showcases how to leverage embeddings, vector databases, GPT-4, and natural language processing techniques to perform robust and efficient information retrieval.

## How to Use

1. Clone the repository and navigate to the notebook directory.
2. Install the required dependencies using:
   ```bash
   pip install -r requirements.txt
3. Add your API keys and relevant paths to the Jupyter notebook before using.
4. Under section 3, there's an import statement happening from a local Python file rather than the native LangChain library. Please be mindful. You can find that file [here](function_calling1.py).

   ```python
   from function_calling1 import _parse_google_docstring
