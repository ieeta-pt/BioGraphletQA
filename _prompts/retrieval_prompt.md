Your role is to analyze the Question Answer pair, as well as the single document provided. 
You need to identify if the document is relevant, and if it is, extract the relevant parts of it, directly quoting the full relevant snippets.

QA: {qa_text}
Document:
ID: {doc['id']}, {doc['text']}

Output format: 
{{
  "documents": [
    {{
      "id": "{doc['id']}",
      "relevant": true/false,
      "snippets": []
    }}
  ]
}}
