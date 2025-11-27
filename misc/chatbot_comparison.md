# Chatbot Performance Comparison: Prompt Engineering vs. RAG

This document provides a complete evaluation of two product inquiry chatbots based on their responses to a common set of 20 user queries.

*   **Chatbot A (PromptE):** An LLM-only chatbot that relies on prompt engineering and has access to product specifications only.
*   **Chatbot B (RAG):** A chatbot that uses a Retrieval-Augmented Generation (RAG) pipeline with access to both product specifications and customer reviews.

## Overall Summary & Final Evaluation

After a comprehensive review of all 20 common queries, the **RAG-based chatbot is the clear winner.**

While the PromptE bot is consistent on simple, factual queries, the RAG bot's ability to synthesize information from both product specifications and customer reviews provides substantially more depth, nuance, and value. It excels at comparative and subjective questions, which are critical for a helpful product assistant. The RAG bot's main weakness is a tendency to fail ungracefully when its retrieval mechanism returns irrelevant documents, as seen in the "TV under $500" query. However, this is an issue with the retrieval/filtering logic rather than the RAG approach itself.

| Metric | PromptE (Non-RAG) | RAG-based | Analysis |
| :--- | :--- | :--- | :--- |
| **Accuracy** | High on simple fact retrieval but often hallucinates or provides incomplete information on complex queries. | Higher overall. Answers are grounded in retrieved documents, leading to more factually dense and nuanced responses. |
| **Completeness** | Often incomplete. It frequently misses products in broad category queries. | Superior. Consistently provides more comprehensive lists of products by retrieving a wider set of relevant documents. |
| **Helpfulness** | Good for basic specs. Fails on any query requiring user opinions or real-world context. | Excellent. Excels at comparative and subjective queries by synthesizing specs and customer sentiment from reviews. |
| **Failure Handling** | Fails gracefully. When it doesn't know, it tends to ask for clarification. | Fails poorly. On one key query, it gave a completely irrelevant answer because its filters returned non-TV products. |
| **Overall Winner** | | 🏆 **RAG** | Despite a significant failure on one query, the RAG bot's ability to provide data-driven, comprehensive, and genuinely helpful answers across a wider range of queries makes it the superior system. |

---

## Detailed Query-by-Query Evaluation

Here is a breakdown of how each chatbot performed on all 20 common queries.

| # | Query | Winner | Analysis |
| - | ----- | ------ | -------- |
| 1 | **What laptops do you have?** | 🏆 **RAG** | **RAG** listed 3 distinct laptop models. **PromptE** missed the gaming laptop, providing an incomplete list. |
| 2 | **Do you have any Gaming laptops?** | 👔 **Tie** | Both bots correctly identified the "BlueWave Gaming Laptop" and provided accurate specs. |
| 3 | **What Lightweight laptops do you have?** | 🏆 **PromptE** | **PromptE** correctly identified two potential options. **RAG** only focused on the "TechPro Ultrabook", whose description explicitly contains "lightweight", making its answer less comprehensive. |
| 4 | **Do you have cameras under $500?** | 🏆 **RAG** | **RAG** provided a more exhaustive list of cameras in the price range (3 options). **PromptE** only identified one. |
| 5 | **What do customers say about battery life of TechPro Ultrabook?** | 🏆 **RAG** | **PromptE** correctly states it has no access to reviews. **RAG** shines here by synthesizing multiple user reviews to give a nuanced answer about battery life. |
| 6 | **What TV under $500 do you have?** | 🏆 **PromptE** | This was a major failure for the RAG bot. **PromptE** incorrectly suggested a TV over budget but correctly identified its mistake. **RAG** failed completely, stating it had no information because its price filter returned irrelevant (non-TV) items. |
| 7 | **What Audio products do you have?** | 🏆 **RAG** | **RAG** listed 5 audio products, providing a more complete list than **PromptE**, which listed only 4. |
| 8 | **Compare GameSphere X and Y.** | 🏆 **RAG** | This query highlights the power of RAG. While **PromptE** gave a good spec comparison, **RAG** provided a far superior answer by incorporating customer feedback from reviews (e.g., lag, buggy multiplayer). |
| 9 | **Do you have any desktop computers?** | 👔 **Tie** | Both bots correctly identified the "TechPro Desktop" and provided accurate specs. |
| 10 | **What televisions do you have that have great display?** | 🏆 **RAG** | **RAG** provided a clean, accurate list of two high-end TVs. **PromptE** became confused, hallucinated a model that isn't in the product list, and incorrectly suggested a home theater system. |
| 11 | **Are the SmartX EarBuds comfortable to wear?** | 🏆 **RAG** | **PromptE** could only guess based on features. **RAG** correctly used the product description ("comfortable earbuds") and customer reviews to directly answer the question. |
| 12 | **I'm looking for a laptop with a good battery life, any recommendations?** | 🏆 **RAG** | **PromptE** hallucinates battery life figures. **RAG** provides a more trustworthy answer by citing what a customer actually reported in a review (6-7 hours). |
| 13 | **Compare the TechPro Ultrabook and the PowerLite Convertible.** | 🏆 **RAG** | Both bots gave good answers, but **RAG** was more comprehensive by including the price difference, a key factor in any comparison. |
| 14 | **What are the features of the ProGamer Racing Wheel?** | 👔 **Tie** | Both bots correctly and completely listed the product's features. |
| 15 | **Show me all smartphones from SmartX.** | 🏆 **PromptE** | **PromptE** provided a more detailed upfront response, including full specs and price for both phones. **RAG**'s answer was correct but less detailed. |
| 16 | **What is the warranty on the CineView 8K TV?** | 👔 **Tie** | Both bots answered the question correctly and concisely. |
| 17 | **What would be a good laptop for a college student?** | 🏆 **RAG** | **PromptE** gives a good, reasoned recommendation. However, **RAG** also recommends the TechPro Ultrabook and grounds its answer in a retrieved review that mentions the product is "perfect for work or school", making its suggestion more evidence-based. |
| 18 | **Can you propose a durable camera for hiking.** | 👔 **Tie** | Both bots correctly identify the ActionCam 4K. **PromptE** mentions "waterproof," while **RAG** retrieves the product based on its "rugged" description. Both are equally helpful. |
| 19 | **I'm a graphic designer looking for a high-performance laptop...** | 🏆 **PromptE** | Both bots correctly identify that there is no specific information on color-accurate displays. However, **PromptE** gives a superior response by asking excellent, targeted clarifying questions about color gamut and portability to better help the user. |
| 20 | **My family loves movie nights. What products would help create an immersive home theater experience?** | 🏆 **RAG** | Both bots recommend a good mix of TVs and audio equipment. **RAG** wins by retrieving a customer review for the SoundMax Home Theater that explicitly mentions it's "perfect for movie nights or gaming sessions", adding valuable social proof to its recommendation. |

---
