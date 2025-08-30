# RAG-PDF-Chatbot
A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask questions about their content. Built with IBM Watsonx, LangChain, and Gradio.

## Features

📄 PDF Upload: Upload any PDF document
🤖 AI-Powered QA: Ask questions about the document content
🔍 Semantic Search: Uses vector embeddings for accurate information retrieval
🌐 Web Interface: User-friendly Gradio interface
⚡ Fast Processing: Efficient document chunking and retrieval

## Technology Stack

LLM: IBM Watsonx (Mixtral-8x7B-Instruct)
Embeddings: IBM Slate-125M English Retriever
Vector Database: Chroma
Framework: LangChain
Interface: Gradio
PDF Processing: PyPDF

## Quick Start
Prerequisites

Python 3.8 or higher
IBM Watsonx AI account and API credentials

## Usage

Upload a PDF: Click on the file upload area and select a PDF document
Ask a Question: Type your question in the text box
Get Answers: Click submit and receive AI-generated answers based on your document

## Configuration
You can modify the following settings in app.py:

Model Parameters: Adjust temperature, max tokens, etc.
Chunk Size: Modify text splitting parameters
Retrieval Settings: Change number of retrieved documents
Server Settings: Modify host and port

## API Configuration
This project uses IBM Watsonx AI services. Make sure you have:

Valid IBM Cloud account
Watsonx AI service instance
Proper API credentials configured

## Contributing

Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add some amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.
Troubleshooting
Common Issues

Authentication Errors: Ensure your Watsonx API credentials are correct
PDF Loading Issues: Check if the PDF file is not corrupted or password-protected
Memory Issues: For large PDFs, consider reducing chunk size or increasing system memory

## Error Messages

Error: Could not load the PDF file - Check file format and integrity
Error creating LLM - Verify API credentials and network connection
Error creating embedding model - Check embedding model availability

## Acknowledgments

IBM Watsonx AI for the language models
LangChain for the RAG framework
Gradio for the web interface
The open-source community for various tools and libraries
