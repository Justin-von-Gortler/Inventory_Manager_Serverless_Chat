/**
 * api/index.js
 *
 * Single-file serverless function (Node) for:
 *  - /api/chat           POST { userId?, message, mode? }
 *  - /api/confirm_action POST { action }  <-- executes DB writes
 *  - /api/image_process  POST { imageUrl }
 *
 * NOTES:
 *  - Uses Hugging Face for text generation (HF_MODEL) and embeddings (HF_EMBEDDING_MODEL).
 *  - Uses Supabase for vector storage & similarity search (adjust SQL/RPC as needed for your setup).
 *  - Uses Airtable REST API for create/update/read operations.
 *
 * Deploy: as a Vercel API route or any serverless function with Node 18+ (fetch available).
 *
 * Fill environment variables before deploying.
 */

// index.js
// Express.js implementation of the chat "brain" using Hugging Face + Supabase + Airtable
// Requires: node >= 16 (18+ recommended). Use "type": "module" in package.json.

import express from 'express';
import fetch from 'node-fetch'; // remove this import if running Node 18+ and prefer global fetch
import { createClient } from '@supabase/supabase-js';

// --- ENV / config
const HF_API_KEY = process.env.HF_API_KEY || '';
const HF_MODEL = process.env.HF_MODEL || 'gpt2'; // override with your HF chat model
const HF_EMBEDDING_MODEL = process.env.HF_EMBEDDING_MODEL || 'sentence-transformers/all-MiniLM-L6-v2';

const SUPABASE_URL = process.env.SUPABASE_URL || '';
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_KEY || '';

const AIRTABLE_KEY = process.env.AIRTABLE_KEY || '';
const AIRTABLE_BASE = process.env.AIRTABLE_BASE || '';
const AIRTABLE_API_VERSION = process.env.AIRTABLE_API_VERSION || 'v0';

if (!HF_API_KEY || !SUPABASE_URL || !SUPABASE_SERVICE_KEY || !AIRTABLE_KEY || !AIRTABLE_BASE) {
  console.warn('Warning: some environment variables are missing. Make sure HF_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY, AIRTABLE_KEY, AIRTABLE_BASE are set.');
}

// --- Supabase client
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);

// --- Airtable helpers
async function airtableFetch(path, options = {}) {
  const url = `https://api.airtable.com/${AIRTABLE_API_VERSION}/${AIRTABLE_BASE}/${path}`;
  const res = await fetch(url, {
    headers: {
      Authorization: `Bearer ${AIRTABLE_KEY}`,
      'Content-Type': 'application/json'
    },
    ...options
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Airtable error ${res.status}: ${text}`);
  }
  return res.json();
}

async function airtableSearch(table, filterFormula, maxRecords = 100) {
  const url = `https://api.airtable.com/${AIRTABLE_API_VERSION}/${AIRTABLE_BASE}/${encodeURIComponent(table)}?filterByFormula=${encodeURIComponent(filterFormula)}&maxRecords=${maxRecords}`;
  const res = await fetch(url, {
    headers: {
      Authorization: `Bearer ${AIRTABLE_KEY}`,
      'Content-Type': 'application/json'
    }
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`Airtable search error: ${txt}`);
  }
  return res.json();
}

async function airtableCreate(table, fields) {
  return airtableFetch(`${encodeURIComponent(table)}`, {
    method: 'POST',
    body: JSON.stringify({ fields })
  });
}

async function airtableUpdate(table, recordId, fields) {
  return airtableFetch(`${encodeURIComponent(table)}/${recordId}`, {
    method: 'PATCH',
    body: JSON.stringify({ fields })
  });
}

// --- Hugging Face helpers
async function hfGenerate(inputs, params = {}) {
  if (!HF_API_KEY) throw new Error('Missing HF_API_KEY env var');
  const modelPath = HF_MODEL;
  const url = `https://api-inference.huggingface.co/models/${modelPath}`;
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${HF_API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ inputs, parameters: params })
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`HF generation error ${res.status}: ${txt}`);
  }
  const out = await res.json();
  if (Array.isArray(out) && out.length > 0 && out[0].generated_text) return out[0].generated_text;
  if (out.generated_text) return out.generated_text;
  if (typeof out === 'string') return out;
  if (out.error) throw new Error(`HF generation error: ${out.error}`);
  return JSON.stringify(out);
}

async function hfEmbed(text) {
  if (!HF_API_KEY) throw new Error('Missing HF_API_KEY env var for embeddings');
  const url = `https://api-inference.huggingface.co/embeddings`;
  const body = { model: HF_EMBEDDING_MODEL, inputs: text };
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${HF_API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(body)
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`HF embedding error ${res.status}: ${txt}`);
  }
  const out = await res.json();
  if (out.embedding) return out.embedding;
  if (Array.isArray(out) && Array.isArray(out[0])) return out[0];
  if (Array.isArray(out) && typeof out[0] === 'number') return out;
  throw new Error('Unexpected embeddings response from HF: ' + JSON.stringify(out).slice(0, 200));
}

// --- Supabase vector search helper
async function supabaseVectorSearch(queryEmbedding, k = 6) {
  // Ensure you created an RPC function named 'match_documents' in Supabase that accepts a vector and a match_count
  const rpcName = 'match_documents';
  const { data, error } = await supabase.rpc(rpcName, {
    query_embedding: queryEmbedding,
    match_count: k
  });
  if (error) {
    throw new Error(`Supabase RPC error: ${error.message}. Ensure you created match_documents RPC.`);
  }
  return data;
}

// --- Prompt
function buildPromptForQA(contextSnippets, userMessage) {
  const context = contextSnippets.slice(0, 8).join('\n');
  return `
You are a helpful assistant that answers user questions about a food/meal/inventory Airtable.
Use only the facts in the "CONTEXT" section to answer. If you need to propose a database change (create or update records), return a JSON object labelled ACTION exactly on its own line after your plain-text answer. The ACTION object must be valid JSON and have the form:
{ "action": "<create_meal|create_inventory|update_inventory|none>", "payload": { ... } }

CONTEXT:
${context}

User question:
${userMessage}

Answer succinctly in human-readable form (max 200 words). If you propose an ACTION, include it as JSON on its own line after the answer.
`;
}

// --- Express app
const app = express();
app.use(express.json({ limit: '6mb' }));

app.get('/', (req, res) => {
  res.json({ ok: true, routes: ['/api/chat (POST)', '/api/confirm_action (POST)', '/api/image_process (POST)'] });
});

// POST /api/chat
app.post('/api/chat', async (req, res) => {
  try {
    const { userId, message, mode } = req.body || {};
    if (!message) return res.status(400).json({ error: 'Missing message' });

    // Resolve user(s) if provided (by Airtable record id or by name)
    let userRecords = [];
    if (userId) {
      if (typeof userId === 'string' && userId.startsWith('rec')) {
        try {
          const userGet = await airtableFetch(`Users/${userId}`);
          if (userGet) userRecords.push(userGet);
        } catch (e) {
          console.warn('Airtable single user fetch failed', e.message);
        }
      } else {
        try {
          const searchRes = await airtableSearch('Users', `{User Name} = "${userId}"`, 5);
          if (searchRes && Array.isArray(searchRes.records)) userRecords = searchRes.records;
        } catch (e) {
          console.warn('Airtable user search failed', e.message);
        }
      }
    }

    // Build context snippets from linked records or vector search fallback
    const contextSnippets = [];
    if (userRecords.length > 0) {
      for (const rec of userRecords) {
        const userName = (rec.fields && (rec.fields['User Name'] || rec.fields['name'])) || 'Unknown user';
        // pull linked Food Classifications
        if (rec.fields && rec.fields['Food Classifications']) {
          const linkedIds = rec.fields['Food Classifications'];
          for (const fid of linkedIds) {
            try {
              const fc = await airtableFetch(`Food_Classifications/${fid}`);
              if (fc && fc.fields) {
                const foodName = (fc.fields['Food'] && Array.isArray(fc.fields['Food']) ? fc.fields['Food'][0] : fc.fields['Food']) || fc.fields['Food Name'] || 'unknown food';
                const classification = fc.fields['Classification'] || fc.fields['classification'] || 'unknown';
                contextSnippets.push(`${userName} classifies ${foodName} as ${classification}`);
              }
            } catch (e) {
              console.warn('Failed to fetch linked food_classification', e.message);
            }
          }
        }

        // optional: fetch inventory items linked to user (if schema supports)
        try {
          const invRes = await airtableSearch('Inventory_Items', `FIND("${userName}", {Owner})`, 20).catch(() => null);
          if (invRes && Array.isArray(invRes.records)) {
            for (const item of invRes.records) {
              const f = item.fields || {};
              const name = f['Food Name'] || f['food_name'] || f['Name'] || 'unknown';
              const qty = f['Quantity'] || f['quantity'] || 'unknown qty';
              const expiry = f['Expiry Date'] || f['expiry_date'] || '';
              contextSnippets.push(`Inventory: ${name} qty:${qty} ${expiry ? `expiry:${expiry}` : ''}`);
            }
          }
        } catch (e) {
          // ignore
        }
      }
    }

    // If context is empty, perform semantic search on Supabase with embedding of message
    if (contextSnippets.length === 0) {
      let qEmbedding = null;
      try {
        qEmbedding = await hfEmbed(message);
      } catch (e) {
        console.warn('Embedding failed:', e.message);
      }
      if (qEmbedding) {
        try {
          const matches = await supabaseVectorSearch(qEmbedding, 6);
          for (const m of matches || []) {
            if (m.content) contextSnippets.push(m.content + (m.metadata ? ` | ${JSON.stringify(m.metadata)}` : ''));
          }
        } catch (e) {
          console.warn('Supabase vector search failed:', e.message);
        }
      } else {
        // Fallback: retrieve a few Food_Classifications records
        try {
          const fc = await airtableFetch('Food_Classifications?maxRecords=10');
          if (fc && Array.isArray(fc.records)) {
            for (const r of fc.records) {
              const f = r.fields || {};
              const foodName = f['Food'] || f['food'] || f['Food Name'] || 'unknown';
              const classification = f['Classification'] || 'unknown';
              contextSnippets.push(`${foodName} classified as ${classification}`);
            }
          }
        } catch (e) { /* ignore */ }
      }
    }

    const prompt = buildPromptForQA(contextSnippets, message);
    const params = { max_new_tokens: 512, temperature: 0.2 };
    const llmOutput = await hfGenerate(prompt, params);

    // Attempt to extract an ACTION JSON at the end
    let action = null;
    const jsonMatch = llmOutput.match(/(\{(?:[\s\S]*\n?)*\})\s*$/);
    if (jsonMatch) {
      try {
        const parsed = JSON.parse(jsonMatch[1]);
        if (parsed && parsed.action) action = parsed;
      } catch (e) {
        console.warn('Failed to parse JSON action from LLM output', e.message);
      }
    }

    res.json({ reply: llmOutput, suggested_action: action });
  } catch (err) {
    console.error('Chat handler error', err);
    res.status(500).json({ error: err.message || String(err) });
  }
});

// POST /api/confirm_action
app.post('/api/confirm_action', async (req, res) => {
  try {
    const action = req.body && req.body.action;
    if (!action || !action.action) return res.status(400).json({ error: 'Missing action object' });

    const act = action.action;
    if (act === 'create_meal') {
      const payload = action.payload || {};
      const created = await createMealRecords(payload);
      return res.json({ ok: true, result: created });
    } else if (act === 'create_inventory') {
      const payload = action.payload || {};
      const created = await createInventoryRecord(payload);
      return res.json({ ok: true, result: created });
    } else if (act === 'update_inventory') {
      const payload = action.payload || {};
      const updated = await updateInventoryRecord(payload);
      return res.json({ ok: true, result: updated });
    } else {
      return res.status(400).json({ error: `Unsupported action type: ${act}` });
    }
  } catch (err) {
    console.error('Confirm action error', err);
    res.status(500).json({ error: err.message || String(err) });
  }
});

// POST /api/image_process
app.post('/api/image_process', async (req, res) => {
  try {
    const { imageUrl } = req.body || {};
    if (!imageUrl) return res.status(400).json({ error: 'Missing imageUrl' });

    // Try HF model for simple caption (vision-capable models only)
    let caption = null;
    try {
      const modelUrl = `https://api-inference.huggingface.co/models/${HF_MODEL}`;
      const genRes = await fetch(modelUrl, {
        method: 'POST',
        headers: { Authorization: `Bearer ${HF_API_KEY}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ inputs: imageUrl })
      });
      if (genRes.ok) {
        const genOut = await genRes.json();
        if (Array.isArray(genOut) && genOut[0] && genOut[0].generated_text) caption = genOut[0].generated_text;
        else if (genOut && genOut.generated_text) caption = genOut.generated_text;
        else if (typeof genOut === 'string') caption = genOut;
      }
    } catch (e) {
      console.warn('HF vision/caption attempt failed:', e.message);
    }

    const queryText = (caption && `${caption}`) || 'food photo';
    let queryEmbedding = null;
    try {
      queryEmbedding = await hfEmbed(queryText);
    } catch (e) {
      console.warn('Embedding failed for image query:', e.message);
    }

    let matches = [];
    if (queryEmbedding) {
      try {
        matches = await supabaseVectorSearch(queryEmbedding, 6);
      } catch (e) {
        console.warn('Supabase vector search failed for image process:', e.message);
      }
    } else {
      // fallback: fetch some Foods records
      try {
        const foods = await airtableFetch('Foods?maxRecords=10');
        if (foods && Array.isArray(foods.records)) {
          matches = foods.records.map(r => ({ id: r.id, content: r.fields['Food Name'] || r.fields['Name'] || '', metadata: r.fields }));
        }
      } catch (e) { /* ignore */ }
    }

    res.json({ imageUrl, caption, matches });
  } catch (err) {
    console.error('Image process error', err);
    res.status(500).json({ error: err.message || String(err) });
  }
});

// --- Helper functions used by /confirm_action

async function createMealRecords(payload) {
  const mealName = payload.meal_name || `Meal ${new Date().toISOString()}`;
  const date = payload.date || new Date().toISOString();
  const userIds = payload.user_ids || [];
  const ingredients = payload.ingredients || [];

  const mealFields = {
    'Meal Name': mealName,
    'Date': date,
    'Users': userIds
  };
  const createdMeal = await airtableCreate('Meals', mealFields);
  const mealRecordId = createdMeal.id;
  const createdIngredients = [];
  for (const ing of ingredients) {
    const fields = {
      'Meal': [mealRecordId],
      'Food': ing.food_id ? [ing.food_id] : [],
      'Food Name': ing.food_name || '',
      'Quantity': ing.quantity || 1
    };
    try {
      const ci = await airtableCreate('Meal_Ingredients', fields);
      createdIngredients.push(ci);
    } catch (e) {
      console.warn('Failed creating ingredient', e.message);
    }
  }
  return { meal: createdMeal, ingredients: createdIngredients };
}

async function createInventoryRecord(payload) {
  const fields = {};
  if (payload.food_id) fields['Food'] = [payload.food_id];
  if (payload.food_name) fields['Food Name'] = payload.food_name;
  if (payload.quantity !== undefined) fields['Quantity'] = payload.quantity;
  if (payload.expiry_date) fields['Expiry Date'] = payload.expiry_date;
  if (payload.ownerUserId) fields['Owner'] = [payload.ownerUserId];
  if (payload.photo_url) fields['Photo'] = [{ url: payload.photo_url }];

  const created = await airtableCreate('Inventory_Items', fields);
  return created;
}

async function updateInventoryRecord(payload) {
  if (!payload.recordId) throw new Error('updateInventoryRecord requires payload.recordId');
  const updated = await airtableUpdate('Inventory_Items', payload.recordId, payload.fields);
  return updated;
}

// Start server if this file is run directly
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Chat brain Express server listening on port ${PORT}`);
});

