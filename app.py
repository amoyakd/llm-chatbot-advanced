"""
Entry point for Hugging Face Spaces deployment.

This file imports and launches the main Gradio chatbot application.
"""

from chatbot_app import demo

if __name__ == "__main__":
    demo.launch()
