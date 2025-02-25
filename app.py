import json
from flask import Flask, request, Response, stream_with_context
from flask_cors import CORS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from openai import OpenAI as OA
import os
from pinecone import Pinecone
import cohere
from pinecone_text.sparse import BM25Encoder
from datetime import datetime
import numpy as np

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt_tab')

bm25_encoder = BM25Encoder().load("bm25_values.json")

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PC_API_KEY")
cohere_api_key = os.getenv("CO_API_KEY")

embedding = OpenAIEmbeddings(model="text-embedding-3-large",api_key=openai_api_key)
client = OA(api_key=openai_api_key)
co = cohere.ClientV2(api_key=cohere_api_key)
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("rbifinal")

GPT_MODEL = "gpt-4o-mini"

# --- Function Definitions ---

def generate_response(prompt, model, json_mode=False, stream=False):
    """
    Calls the OpenAI chat API.
    (If your API/client supports streaming, pass stream=True; here we use a non-streaming call.)
    """
    
    if json_mode:
        response = client.chat.completions.create(
            temperature=0.1,
            model=model,
            messages=[{
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }],
            response_format={"type": "json_object"},
            stream=stream
        )
    else:
        response = client.chat.completions.create(
            temperature=0.1,
            model=model,
            messages=[{
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }],
            stream=stream
        )
    return response

def retrieval(query,index,bm25_encoder,embedding, k):

    sparce_query_vector = bm25_encoder.encode_queries(query)
    dense_query_vector = embedding.embed_query(query)

    query_response = index.query(
    top_k=k,
    vector=dense_query_vector,
    sparse_vector=sparce_query_vector,
    include_metadata=True)   
    
    return query_response

def compute_confidence_hybrid(results, lambda_=0.5, alpha=0.3, min_boost_threshold=0.5, boost_factor=0.1):

    scores = np.array([item.relevance_score for item in results.results])
    weights = np.exp(-lambda_ * np.arange(len(scores)))
    C_exp_decay = np.sum(scores * weights) / np.sum(weights)
    top_k = min(3, len(scores))
    C_top5_avg = np.mean(scores[:top_k])
    confidence_score = (alpha * C_exp_decay) + ((1 - alpha) * C_top5_avg)
    num_moderate_docs = np.sum(scores > min_boost_threshold)
    if num_moderate_docs >= 1: 
        confidence_score += boost_factor
    confidence_score = min(confidence_score, 1.0)

    return confidence_score

def reranking(co,query,query_response,k1):

    results = co.rerank(
    model="rerank-v3.5",
    query=query,
    documents=[resp.metadata['context'] for resp in query_response['matches']],
    top_n=k1)

    final_docs = [
    Document(
        page_content=(
            m := query_response["matches"][i.index]["metadata"].copy()
        ).pop("original", ""),
        metadata={k: v for k, v in m.items() if k != "context"}
    )
    for i in results.results]

    confidence = compute_confidence_hybrid(results, lambda_=0.5)

    return confidence,final_docs


def hyde_response_final(query, GPT_MODEL):
    prompt =  """\nQUERY: """ + str(query) + """\nINSTRUCTIONS:
    You are an AI assistant (LLM #1) specializing in Indian banking and financial compliance queries. 
    Your goals are to:

    1. **Classify** the user's query into one of the following categories:
        - **Informational**: The user wants to learn or understand a concept, update, or explanation.
        - **Navigational**: The user is trying to locate a specific resource (e.g., a document, URL, or regulation section).
        - **Transactional**: The user wants to perform a task or produce an artifact (e.g., generate a checklist, compare documents, or analyze user-provided information).
        - **Vague**: Too broad or ambiguous. These need user clarification.
        - **Irrelevant**: Completely Off-topic (e.g., unrelated to banking, finance, or compliance).
    2. **Refine** the user's query to improve retrieval. 
        - You may return **a single refined query** or **a list of refined queries** if that could produce better search results.
        - For **Informational queries**, expand the query into detailed, specific, and regulatory-focused search queries that are similar in meaning by incorporating relevant compliance terminology, full-forms,  synonyms, and related concepts. Consider adding relevant keywords, full forms or clarifications (e.g., specific regulations, time periods) if they are implied by context.
        - For **Navigational queries**, ensure precision: reference exact document names, sections, or resource identifiers.
        - For **Transactional queries**:
            - If the user provides additional custom inputs or data (e.g., sample text, internal policy info), ensure you **incorporate** that information into the refinement so that a retrieval system can locate or match relevant external documents.
            - Clarify the user’s requested “action” (e.g., generate a checklist, perform a gap analysis, create a comparison table, etc.).

    3. **Output** must be valid JSON **without extra commentary**. 
        - Use the following structure (note the possibility of multiple refined queries):
        { "category": "<Informational | Navigational | Transactional | Vague | Irrelevant>", "refinedQuery": [ "<string of refined query #1>", "<string of refined query #2>", ... ] }
    -------------------------------------------------------------------------
    Refer to these few-shot examples to guide your output formatting:

    -----
    **Example 1 (Informational)**
    User: "What are the new RBI guidelines for AML compliance this year?"
    Output:
    {
    "category": "Informational",
    "refinedQuery": [
    "Retrieve the most recent RBI guidelines on Anti-money laundering (AML) compliance under the Prevention of Money Laundering Act (PMLA) in 2025."
    ]
    }

    **Example 2 (Navigational)**
    User: "Where can I find the RBI 2024 Master Directions on KYC?"
    Output:
    {
    "category": "Navigational",
    "refinedQuery": [
    "Locate the official RBI Master Directions on KYC (2024 edition) PDF"
    ]
    }

    **Example 3 (Transactional)**
    User: "Generate a checklist of key RBI compliance obligations for banks in India."
    Output:
    {
    "category": "Transactional",
    "refinedQuery": [
    "Generate a compliance checklist for key RBI regulatory obligations applicable to banks in India."
    ]
    }

    **Example 4 (Transactional + user input)** 
    User: "Compare my bank’s internal policy on high-risk customer onboarding with current RBI KYC & PMLA guidelines"
    Output:
    {
    "category": "Transactional",
    "refinedQuery": [
    "Perform a gap analysis comparing the bank’s internal policy (provided in user text) with RBI Master Directions on KYC and PMLA guidelines."
    ]
    }

    **Example 5 (Multiple refined queries)** 
    User: "I want to compare AML compliance changes from RBI since 2023. Also, check if there are any new guidelines on KYC compliance?"
    Output:
    {
    "category": "Informational",
    "refinedQuery": [
    "Retrieve recent AML compliance changes from RBI since 2023.",
    "Retrieve the latest RBI guidelines on KYC compliance."
    ]
    }

    **Example 6 (Vague or incomplete query)** 
    User: "What is the process for approval?"
    Output:
    {
    "category": "Vague",
    "refinedQuery": []
    }

    **Example 7 (Irrelevant)** 
    User: "Who is your favorite footballer?"
    Output:
    {
    "category": "Irrelevant",
    "refinedQuery": []
    }
    -----

    IMPORTANT:
    • Return only valid JSON, no extra commentary.
    • Do not include text outside the JSON response.
    • Include full-forms wherever possible
    • Today's date is """ +datetime.today().strftime("%B %d, %Y")+""". The retrieval system contains up-to-date information. 
    • Do not assume outdated timeframes like "as of 2023" unless explicitly mentioned by the user.
    """
    response = generate_response(prompt=prompt, model="gpt-4o", json_mode=True)
    return response

# --- Flask App ---

app = Flask(__name__)

CORS(app)

@app.route('/query', methods=['POST'])
def query_endpoint():
    data = request.get_json()
    query_input = data.get("query")
    if not query_input:
        return Response("No query provided", status=400)
    
    
    @stream_with_context
    def generate():
    
        hyde_resp = hyde_response_final(query_input, GPT_MODEL)
        hyde_json = json.loads(hyde_resp.choices[0].message.content)
        if (hyde_json['category']=='Vague')|(hyde_json['category']=='Irrelevant'):
            rejection_prompt=f"""You are an AI assistant specializing in **Indian banking and financial compliance**.  
            You will receive a **user query** along with its **classification** ("Vague" or "Irrelevant").
            Your task is to generate a **polite, professional, and context-aware response**:  
            ["userQuery": {query_input}, "classification": {hyde_json['category']}]

            ---

            ### **If the Query is Vague**  
            - **Do NOT reject the query.** Instead, guide the user with **up to 3 clarifying questions** to refine their request.  
            - Assume they are asking about **banking, finance, or compliance**, but may not have phrased it well.  
            - Format the response as:  
            1. Acknowledging their query.  
            2. Providing **up to 3 bullet-pointed follow-up questions** to help narrow down their intent.  

            **Example:**  
            _User Query:_ `"Tell me about regulations."`  
            _Response:_  
            *"Regulations cover a wide range of banking and compliance areas. Could you clarify what you're looking for? Here are a few guiding questions:"*  
            - **Are you referring to RBI regulations, SEBI guidelines, or any specific regulatory body?**  
            - **Do you need compliance requirements for banks, NBFCs, or fintech companies?**  
            - **Are you looking for recent regulatory updates or general compliance frameworks?**  

            ---

            ### **If the Query is Irrelevant**  
            - Politely inform the user that you specialize in **Indian banking and financial compliance**.  
            - Keep the tone **courteous and professional** without sounding dismissive.  
            - Encourage them to ask about relevant topics instead.  

            **Example:**  
            _User Query:_ `"Who is your favorite footballer?"`  
            _Response:_  
            *"I focus on banking and financial compliance topics. Let me know if you have any questions related to RBI regulations, KYC, AML, or other compliance matters!"*  

            ---

            ### **Few-Shot Examples**
            #### **Example 1: Vague Query**
            **User Query:** `"What is the process for approval?"`  
            **Expected Response:**  
            *"Approval processes vary based on context. Could you clarify which approval you're referring to?"*  
            - **Are you asking about RBI approvals for new banking products or internal compliance approvals?**  
            - **Do you need information on loan approval regulations or customer onboarding processes?**  
            - **Are you looking for a checklist or a step-by-step guideline?**  

            #### **Example 2: Irrelevant Query**
            **User Query:** `"Write a Python script to create your replica."`  
            **Expected Response:**  
            *I'm here to assist with banking and financial compliance topics. If you have questions about RBI regulations, compliance workflows, or regulatory updates, I’d be happy to help!*  

            #### **Example 3: Another Vague Query**
            **User Query:** `"I need details on policy framework."`  
            **Expected Response:**  
            *"There are multiple policy frameworks in banking and finance. Could you clarify what you’re referring to?"*  
            - **Are you asking about RBI's regulatory policies or internal bank policies?**  
            - **Do you need compliance guidelines for KYC, AML, or another specific area?**  
            - **Are you looking for a comparison between different policy frameworks?**  

            ---
            """
            rejection_response = client.chat.completions.create(
            temperature=0.1,
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [{"type": "text", "text": rejection_prompt}]
            }],
            stream=True
            )
            for chunk in rejection_response:
                if chunk.choices[0].delta.content is not None:
                    yield f"data: {json.dumps({'response': chunk.choices[0].delta.content,'type': 'explanation'})}\n\n"
        
        else:

            expanded_query = query_input + ", " + query_input + ", " + ', '.join(hyde_json["refinedQuery"])

            filtered_docs = retrieval(expanded_query,index,bm25_encoder,embedding, k=80)

            if hyde_json["category"] in ["Informational", "Navigational"]:
                k1 = 10
            else:
                k1 = 15

            confidence,final_docs = reranking(co,expanded_query, filtered_docs, k1)

            yield f"data: {json.dumps({'response': str(confidence),'type': 'confidence'})}\n\n"

            # Step 3: Build an explanation prompt based on the query category.
            if hyde_json["category"] == "Informational":
                explanation_prompt = f"""You are a compliance expert assistant specializing in Indian banking and financial compliance. The user’s query is informational, meaning they want a summary or explanation. 
                Use the retrieved documents to provide a clear, concise answer, focusing on key points and relevant details. 
                Here is the user’s query and relevant context (updated as of {datetime.today().strftime("%B %d, %Y")}):
                ["userQuery": {query_input}, "retrievedDocuments": {final_docs}]

                Please format your final answer as a short explanation or summary. Include any critical details such as new rules, regulation names, or timelines.
                Remember: 
                1. Do NOT answer questions completely unrelated to banking or compliance, even if they are well-formed. If the user asks something unrelated, respond that you can only assist with compliance matters.  
                2. Stick to the retrievedDocuments content. If relevant information is found in the documents, use that exclusively. If the retrievedDocuments did not return relevant information, be transparent about it.
                3. Prioritize the latest time frame when selecting relevant regulations. Always refer to the most recent regulations as of {datetime.today().strftime("%B %d, %Y")}
                4. Always prioritize the most recent document. If multiple documents have relevant information:
                    - (a) Look for the answer first in Acts, Act Rules and Regulations. Then look for answer in Master Directions.
                    - (b) If the answer is not found in Master Directions, then look in Master Circulars, Notifications, or Draft Notifications (only if published after the Master Direction).
                    - (c) Draft Notifications should be used last because they are not finalized guidelines.
                """
            elif hyde_json["category"] == "Navigational":
                explanation_prompt = f"""You are a compliance expert assistant specializing in Indian banking and financial compliance. The user wants to locate a specific resource or document. 
                Provide them with direct references, links, or sections. Keep it concise and ensure they can find the material easily. 
                Here is the user’s query and retrieved context (updated as of {datetime.today().strftime("%B %d, %Y")}):
                ["userQuery": {query_input}, "retrievedDocuments": {final_docs}]

                Output your answer as direct references or URLs wherever possible.
                Remember: 
                1. Do NOT answer questions completely unrelated to banking or compliance, even if they are well-formed. If the user asks something unrelated, respond that you can only assist with compliance matters.  
                2. Stick to the retrievedDocuments content. If relevant information is found in the documents, use that exclusively. If the retrievedDocuments did not return relevant information, be transparent about it.
                3. Prioritize the latest time frame when selecting relevant regulations. Always refer to the most recent regulations as of {datetime.today().strftime("%B %d, %Y")}
                4. Always prioritize the most recent document. If multiple documents have relevant information:
                    - (a) Look for the answer first in Acts, Act Rules and Regulations. Then look for answer in Master Directions.
                    - (b) If the answer is not found in Master Directions, then look in Master Circulars, Notifications, or Draft Notifications (only if published after the Master Direction).
                    - (c) Draft Notifications should be used last because they are not finalized guidelines.
                5. Always provide references and links from the original documents whenever possible.  
                """
            elif hyde_json["category"] == "Transactional":
                explanation_prompt = f"""You are a compliance expert assistant. The user wants to perform a task (e.g., generate a checklist, compare regulations). 
                Use the retrieved documents to produce the requested output. 
                If they want a comparison, provide a table; if they want a checklist, provide bullet points; if they want calculations, perform calculations accurately etc. 
                Here is the user’s query and retrieved context (updated as of {datetime.today().strftime("%B %d, %Y")}):
                ["userQuery": {query_input}, "expandedQuery": {expanded_query}, "retrievedDocuments": {final_docs}]

                Remember: 
                1. Do NOT answer questions completely unrelated to banking or compliance, even if they are well-formed. If the user asks something non-compliance-related, respond that you can only assist with compliance matters.  
                2. Stick to the retrievedDocuments content. If relevant information is found in the documents, use that exclusively. If the retrievedDocuments did not return relevant information, be transparent about it.
                3. Prioritize the latest time frame when selecting relevant regulations. Always refer to the most recent regulations as of {datetime.today().strftime("%B %d, %Y")}
                4. Structure your response based on the task type:
                    - If the user requests a checklist, provide clear bullet points**.  
                    - If the user requests a comparison, present information in a table format.  
                    - If the user requests a calculation, ensure accuracy using provided regulatory formulas or standards.
                5. Always prioritize the most recent document. If multiple documents have relevant information:
                    - (a) Look for the answer first in Acts, Act Rules and Regulations. Then look for answer in Master Directions.
                    - (b) If the answer is not found in Master Directions, then look in Master Circulars, Notifications, or Draft Notifications (only if published after the Master Direction).
                    - (c) Draft Notifications should be used last because they are not finalized guidelines.
                6. Always provide references and links from the original documents whenever possible. 
                """
            else:
                explanation_prompt = "Invalid category."
            
            explanation_response = client.chat.completions.create(
                temperature=0.1,
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [{"type": "text", "text": explanation_prompt}]
                }],
                stream=True
            )
            for chunk in explanation_response:
                if chunk.choices[0].delta.content is not None:
                yield f"data: {json.dumps({'response': chunk.choices[0].delta.content,'type': 'explanation'})}\n\n"


            prompt_documents = ""
            for i, doc in enumerate(final_docs, start=1):
                doc_name = doc.metadata.get("name", "Unknown Name")
                doc_date = doc.metadata.get("date", "Unknown Date")
                doc_type = doc.metadata.get("type", "Unknown Type")
                doc_link = doc.metadata.get("link", "Unknown link")
                
                prompt_documents += f"DOCUMENT #{i}\n"
                prompt_documents += f"Document Name: {doc_name}\n"
                prompt_documents += f"Document Date: {doc_date}\n"
                prompt_documents += f"Document Link: {doc_link}\n"
                prompt_documents += f"Document Type: {doc_type}\n\n"
                prompt_documents += f"{doc.page_content}\n\n"

            reference_prompt = f"""
            SYSTEM NOTE:
            You are an automated text-extraction system. You must follow the INSTRUCTIONS strictly. 
            Do not add extra content, explanations, or summaries beyond what the INSTRUCTIONS permit.

            DOCUMENT(S):
            {prompt_documents}

            QUESTION:
            {query_input}

            INSTRUCTIONS:
            1. Read the DOCUMENT(S) and extract only the part(s) which directly relevant to the QUESTION. 
                - Provide sufficient context (i.e., the entire paragraph) so the meaning is clear.

            2. Always prioritize the most recent document. If multiple documents have relevant information:
                - (a) Look for the answer first in Acts, Act Rules and Regulations. Then look for answer in Master Directions.
                - (b) If the answer is not found in Master Directions, then look in Master Circulars, Notifications, or Draft Notifications (only if published after the Master Direction).
                - (c) Draft Notifications should be used last because they are not finalized guidelines.

            3. Include any necessary exceptions, special conditions, or extended clauses from the DOCUMENT(S) if they are relevant to the QUESTION.

            4. Provide the relevant text exactly as it appears in the DOCUMENT(S). Return whole paragraphs, but highlight (e.g. with **text**) the part that directly addresses the QUESTION.

            5. Do NOT add your own answer, finding, interpretation, summary, or explanation. 
                - There is a separate LLM which will interpret your extracted output.

            6. For each relevant extraction, precede it with the DOCUMENT reference details and the exact section/clauses if available. 
                Example:
                *1. [Document Name](Document Link) (Master Directions) issued on [Date], Section 2.1.1.1:*
                [Extracted Paragraph(s)]

                *2. [Document Name](Document Link) (Notification) issued on [Date], Section 3.2.4:*
                [Extracted Paragraph(s)]

            7. Output the extracted information in descending chronological order (most recent to oldest).

            8. If the DOCUMENT(S) do not contain any relevant information or the QUESTION is completely unrelated to compliance, respond with exactly:
                (No Reference Found)

            9. Produce no text other than:
                - The references of the DOCUMENT(S)
                - The directly relevant paragraphs extracted verbatim (with highlighting)
                - Or "(No Reference Found)" if no relevant info is found.

            END OF INSTRUCTIONS
            """.strip()
        
            # Step 5: Generate references (also streamed from OpenAI).
            reference_response = client.chat.completions.create(
                temperature=0.1,
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [{"type": "text", "text": reference_prompt}]
                }],
                stream=True
            )
            for chunk in reference_response:
                if chunk.choices[0].delta.content is not None:
                    yield f"data: {json.dumps({'response': chunk.choices[0].delta.content,'type': 'reference'})}\n\n"
    return Response(stream_with_context(generate()), content_type="text/event-stream")

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
