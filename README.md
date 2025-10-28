# Inventory_Manager_Serverless_Chat
ready-to-deploy single-file serverless function (Node.js) that implements core server “chat brain” endpoints for integration with Nutrition_Management Airflow table

## Env Vars
- **HF_API_KEY** — Hugging Face inference/embeddings API key

- **HF_MODEL** — Hugging Face model id to use for generation (e.g., tiiuae/falcon-7b-instruct or other HF chat model you prefer)

- **HF_EMBEDDING_MODEL** — Hugging Face embedding model id (e.g., sentence-transformers/all-MiniLM-L6-v2) — or leave blank to reuse generation model if it supports embeddings

- **SUPABASE_URL** — your Supabase project URL

- **SUPABASE_SERVICE_KEY** — Supabase service_role or anon with the proper RBAC for vector table reads/writes

- **AIRTABLE_KEY** — your Airtable API key

- **AIRTABLE_BASE** — your Airtable base id

- **AIRTABLE_API_VERSION** — optional, default v0
