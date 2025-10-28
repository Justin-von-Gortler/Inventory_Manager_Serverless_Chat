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

import fetch from 'node-fetch';
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

// basic validation
if (!HF_API_KEY || !SUPABASE_URL || !SUPABASE_SERVICE_KEY || !AIRTABLE_KEY || !AIRTABLE_BASE) {
  // If running locally, we allow but warn. In production you must set them.
  console.warn('Warning: Some environment variables are not set. Please set HF_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY, AIRTABLE_KEY, AIRTABLE_BASE');
}

// --- Supabase client
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);

// --- Helper: Airtable REST helpers
const airtableFetch = async (path, options = {}) => {
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
};

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

// --- Helper: Hugging Face LLM (text generation)
// This function calls Hugging Face Inference API for generation.
// We keep it generic: pass `inputs` (string) and optional `params` object per HF model.
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
  // HF returns different shapes depending on model; commonly { generated_text } or array of outputs.
  // We try to pull out a suitable text.
  if (Array.isArray(out) && out.length > 0 && out[0].generated_text) return out[0].generated_text;
  if (out.generated_text) return out.generated_text;
  if (typeof out === 'string') return out;
  // If it's an object with 'error', bubble it
  if (out.error) throw new Error(`HF generation error: ${out.error}`);
  // fallback: stringify
  return JSON.stringify(out);
}

// --- Helper: Hugging Face embeddings
// Many HF models expose embeddings via the "embeddings" endpoint. We'll call the standard embeddings endpoint:
// POST https://api-inference.huggingface.co/embeddings with body { model: <model>, inputs: <text> }
// NOTE: If your account / HF model doesn't support embeddings that way, you may need a different flow.
async function hfEmbed(text) {
  if (!HF_API_KEY) throw new Error('Missing HF_API_KEY env var for embeddings');
  // Use explicit embeddings endpoint
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
  // Expect out.embedding or an array. Many HF embedding endpoints return {embedding: [...] } or [[...]]
  if (out.embedding) return out.embedding;
  if (Array.isArray(out) && out.length > 0 && Array.isArray(out[0])) return out[0];
  // fallback: if out is a single numeric array
  if (Array.isArray(out) && typeof out[0] === 'number') return out;
  throw new Error('Unexpected embeddings response from HF: ' + JSON.stringify(out).slice(0, 200));
}

// --- Helper: Supabase vector search
/**
 * NOTE: Supabase vector search depends on how you created your table and the pgvector extension.
 * This function assumes you created a `documents` table with columns:
 *  - id (uuid or int)
 *  - content (text)
 *  - embedding (vector)  -- using pgvector
 *  - metadata (jsonb) optional
 *
 * Two common ways to search:
 *  - Use a SQL query: SELECT *, embedding <-> query_embedding AS distance FROM documents ORDER BY distance LIMIT k;
 *  - Use a user-defined RPC that accepts embeddings as float8[] and performs the query.
 *
 * In supabase-js, you can run raw SQL via supabase.rpc(...) if you created an RPC function; or you can use
 * supabase.from('documents').select(...).limit(k).order('embedding', {ascending: true}) if you have helper operators.
 *
 * Here we'll use a raw SQL rpc approach: you can create a Postgres RPC function in Supabase like:
 *
 * CREATE FUNCTION match_documents(query_embedding vector, match_count int)
 * RETURNS TABLE(id uuid, content text, metadata jsonb, distance float) AS $$
 *   SELECT id, content, metadata, embedding <-> query_embedding AS distance FROM documents ORDER BY distance LIMIT match_count;
 * $$ LANGUAGE SQL STABLE;
 *
 * Then we call `supabase.rpc('match_documents', { query_embedding, match_count: 5 })`
 *
 * If you don't have an RPC, adapt this function to your setup.
 */
async function supabaseVectorSearch(queryEmbedding, k = 6) {
  // Basic validation
  if (!SUPABASE_SERVICE_KEY || !SUPABASE_URL) throw new Error('Supabase env not configured');

  // The RPC name 'match_documents' is an example; create it in your Supabase SQL editor (see comment above).
  // If you used a different name, change it here.
  const rpcName = 'match_documents';

  // Supabase-js expects JS arrays for vector params
  const { data, error } = await supabase.rpc(rpcName, {
    query_embedding: queryEmbedding,
    match_count: k
  });

  if (error) {
    // If the RPC doesn't exist, this will throw. Provide a helpful message.
    throw new Error(`Supabase RPC error: ${error.message}. Ensure you created an RPC match_documents(query_embedding vector, match_count int) that returns id, content, metadata, distance.`);
  }

  return data; // array of rows {id, content, metadata, distance}
}

// --- Prompt templates
function buildPromptForQA(contextSnippets, userMessage) {
  // contextSnippets: array of short strings (e.g., "Alice classifies peanut butter as enjoy")
  const context = contextSnippets.slice(0, 8).join('\n'); // limit snippets
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

// --- Core endpoint implementations

// Handler router: for Vercel / Netlify simple API route style:
export default async function handler(req, res) {
  try {
    if (req.method === 'POST' && req.url.endsWith('/chat')) {
      return await handleChat(req, res);
    } else if (req.method === 'POST' && req.url.endsWith('/confirm_action')) {
      return await handleConfirmAction(req, res);
    } else if (req.method === 'POST' && req.url.endsWith('/image_process')) {
      return await handleImageProcess(req, res);
    } else {
      // Provide a small index/help
      if (req.method === 'GET') {
        res.status(200).json({ ok: true, routes: ['/api/chat (POST)', '/api/confirm_action (POST)', '/api/image_process (POST)'] });
        return;
      }
      res.status(404).json({ error: 'Not found' });
    }
  } catch (err) {
    console.error('Unhandled error', err);
    res.status(500).json({ error: err.message || String(err) });
  }
}

// --- Handler: /api/chat
/**
 * Expected body:
 * {
 *   userId?: "<Airtable user id or name>",
 *   message: "What snacks can Alice and Bob share?",
 *   mode?: "query" | "create_meal" | "inventory_update"
 * }
 */
async function handleChat(req, res) {
  const body = await parseJson(req);
  const { userId, message, mode } = body || {};
  if (!message) return res.status(400).json({ error: 'Missing message' });

  // 1) Attempt to resolve any user names mentioned to Airtable records (simple approach: search by name)
  // For speed, we'll try a crude parse: if user provides userId as a name string, search Airtable Users table
  let userRecords = [];
  if (userId) {
    // If userId looks like an Airtable record id (rec...), we'll skip search and fetch directly
    if (typeof userId === 'string' && userId.startsWith('rec')) {
      const userGet = await airtableFetch(`Users/${userId}`);
      userRecords = userGet ? [userGet] : [];
    } else {
      // Search by name
      const searchRes = await airtableSearch('Users', `{User Name} = "${userId}"`, 5).catch(e => {
        console.warn('Airtable search error', e);
        return null;
      });
      if (searchRes && Array.isArray(searchRes.records)) userRecords = searchRes.records;
    }
  }

  // 2) Gather RAG context: if user mentioned specific users, fetch their linked food_classifications & inventory
  const contextSnippets = [];
  if (userRecords.length > 0) {
    for (const rec of userRecords) {
      const userName = (rec.fields && (rec.fields['User Name'] || rec.fields['name'])) || 'Unknown user';
      // fetch linked food_classifications (Airtable linked record ids -> we must retrieve)
      if (rec.fields && rec.fields['Food Classifications']) {
        const linkedIds = rec.fields['Food Classifications'];
        // fetch each linked food_classifications record
        for (const fid of linkedIds) {
          try {
            const fc = await airtableFetch(`Food_Classifications/${fid}`);
            if (fc && fc.fields) {
              const foodName = (fc.fields['Food'] && Array.isArray(fc.fields['Food']) ? fc.fields['Food'][0] : fc.fields['Food']) || (fc.fields['food_name']) || 'unknown food';
              const classification = fc.fields['Classification'] || fc.fields['classification'] || 'unknown';
              contextSnippets.push(`${userName} classifies ${foodName} as ${classification}`);
            }
          } catch (e) {
            console.warn('Failed to fetch linked fc', e);
          }
        }
      }

      // fetch inventory items that are linked to this user (if your schema links inventory to user)
      // If inventory isn't linked to user, you can skip or do a global inventory fetch instead.
      try {
        // Example: if inventory items have a "Owner" or "User" field linked to Users
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

  // If context is empty, we can perform a vector search on the raw message to return top docs from supabase
  if (contextSnippets.length === 0) {
    // Compute embedding for the user message
    let qEmbedding = null;
    try {
      qEmbedding = await hfEmbed(message);
    } catch (e) {
      console.warn('Embedding failed:', e.message);
    }
    if (qEmbedding) {
      try {
        const matches = await supabaseVectorSearch(qEmbedding, 6);
        // matches expected to contain content and metadata
        for (const m of matches || []) {
          if (m.content) contextSnippets.push(m.content + (m.metadata ? ` | ${JSON.stringify(m.metadata)}` : ''));
        }
      } catch (e) {
        console.warn('Supabase vector search failed:', e.message);
      }
    } else {
      // fallback: fetch some recent food_classifications or inventory items for general context
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

  // 3) Build prompt and call HF LLM
  const prompt = buildPromptForQA(contextSnippets, message);

  // HF generation parameters — you can tune these per your model
  const params = { max_new_tokens: 512, temperature: 0.2 };

  const llmOutput = await hfGenerate(prompt, params);

  // 4) Try to detect an ACTION JSON on its own line at the end of the output
  let action = null;
  // find last JSON block in the output
  const jsonMatch = llmOutput.match(/(\{(?:[\s\S]*\n?)*\})\s*$/);
  if (jsonMatch) {
    try {
      const parsed = JSON.parse(jsonMatch[1]);
      // validate shape
      if (parsed && parsed.action) action = parsed;
    } catch (e) {
      // ignore parse error — we will just return text
      console.warn('Failed to parse action JSON from LLM output', e.message);
    }
  }

  // Return to client: text reply + optional suggested action (not executed)
  res.status(200).json({
    reply: llmOutput,
    suggested_action: action
  });
}

// --- Handler: /api/confirm_action
/**
 * Body: { action: { action: "...", payload: {...} } }
 * Example action for create_meal:
 * {
 *   action: "create_meal",
 *   payload: {
 *     meal_name: "Dinner 2025-10-28",
 *     date: "2025-10-28",
 *     user_ids: ["recXXXX", "recYYYY"],
 *     ingredients: [
 *       { food_name: "Pizza", food_id: "recAAA", quantity: 2 },
 *       ...
 *     ]
 *   }
 * }
 */
async function handleConfirmAction(req, res) {
  const body = await parseJson(req);
  const action = body && body.action;
  if (!action || !action.action) return res.status(400).json({ error: 'Missing action object' });

  // Route by action type
  const act = action.action;
  try {
    if (act === 'create_meal') {
      const payload = action.payload || {};
      const created = await createMealRecords(payload);
      return res.status(200).json({ ok: true, result: created });
    } else if (act === 'create_inventory') {
      const payload = action.payload || {};
      const created = await createInventoryRecord(payload);
      return res.status(200).json({ ok: true, result: created });
    } else if (act === 'update_inventory') {
      const payload = action.payload || {};
      const updated = await updateInventoryRecord(payload);
      return res.status(200).json({ ok: true, result: updated });
    } else {
      return res.status(400).json({ error: `Unsupported action type: ${act}` });
    }
  } catch (err) {
    console.error('Confirm action error', err);
    res.status(500).json({ error: err.message || String(err) });
  }
}

// Helper: createMealRecords(payload)
async function createMealRecords(payload) {
  // Minimal example: create a meal in `Meals` table, and create associated `Meal_Ingredients`.
  const mealName = payload.meal_name || `Meal ${new Date().toISOString()}`;
  const date = payload.date || new Date().toISOString();
  const userIds = payload.user_ids || []; // array of Airtable user record ids
  const ingredients = payload.ingredients || []; // array of { food_id, food_name, quantity }

  // 1) create meal record
  const mealFields = {
    'Meal Name': mealName,
    'Date': date,
    'Users': userIds // assuming Users is a linked-record field in Meals
  };
  const createdMeal = await airtableCreate('Meals', mealFields);

  // 2) create meal_ingredients and link to meal
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

// Helper: createInventoryRecord(payload)
async function createInventoryRecord(payload) {
  // payload: { food_id?, food_name, quantity, expiry_date?, photo_url? , ownerUserId? }
  const fields = {};
  if (payload.food_id) fields['Food'] = [payload.food_id];
  if (payload.food_name) fields['Food Name'] = payload.food_name;
  if (payload.quantity !== undefined) fields['Quantity'] = payload.quantity;
  if (payload.expiry_date) fields['Expiry Date'] = payload.expiry_date;
  if (payload.ownerUserId) fields['Owner'] = [payload.ownerUserId];
  if (payload.photo_url) {
    // Airtable attachment expects an array of objects with url
    fields['Photo'] = [{ url: payload.photo_url }];
  }

  const created = await airtableCreate('Inventory_Items', fields);
  return created;
}

// Helper: updateInventoryRecord(payload)
async function updateInventoryRecord(payload) {
  // payload: { recordId, fields: { Quantity: 5, 'Expiry Date': '2025-11-01' } }
  if (!payload.recordId) throw new Error('updateInventoryRecord requires payload.recordId');
  const updated = await airtableUpdate('Inventory_Items', payload.recordId, payload.fields);
  return updated;
}

// --- Handler: /api/image_process
/**
 * Expects { imageUrl: "https://..." }
 * Returns candidate matches from supabase + optional HF vision caption
 */
async function handleImageProcess(req, res) {
  const body = await parseJson(req);
  const { imageUrl } = body || {};
  if (!imageUrl) return res.status(400).json({ error: 'Missing imageUrl' });

  // Option 1: Use HF model to generate a caption or tags from image (if your HF model supports vision)
  // Many HF vision models accept image bytes via the same model endpoint.
  // We'll attempt a generic call to the chosen HF model; if it doesn't handle images, this will error.
  let caption = null;
  try {
    // Some HF models accept { inputs: image_url } or { inputs: { image: image_url } } - this is model dependent.
    // We send a minimal request; modify for your chosen vision model.
    const modelUrl = `https://api-inference.huggingface.co/models/${HF_MODEL}`;
    const genRes = await fetch(modelUrl, {
      method: 'POST',
      headers: { Authorization: `Bearer ${HF_API_KEY}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ inputs: imageUrl })
    });
    if (genRes.ok) {
      const genOut = await genRes.json();
      // attempt to extract a string caption
      if (Array.isArray(genOut) && genOut[0] && genOut[0].generated_text) caption = genOut[0].generated_text;
      else if (genOut && genOut.generated_text) caption = genOut.generated_text;
      else if (typeof genOut === 'string') caption = genOut;
      else if (genOut && genOut.error) caption = null;
    }
  } catch (e) {
    console.warn('HF vision/caption attempt failed (this may be normal if your HF_MODEL is not vision-capable):', e.message);
  }

  // 2) Build a query phrase from the caption (fallback to 'food photo' if caption missing)
  const queryText = (caption && `${caption}`) || 'food photo';

  // 3) embed the query and search Supabase vector store for best matches in `foods` table
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
    // fallback: do a naive Airtable name search by caption words
    if (caption) {
      const words = caption.split(/\s+/).slice(0, 6).map(w => w.replace(/[^\w]/g,'')).filter(Boolean);
      const formula = words.map(w => `FIND("${w}", {Food Name})`).join(',');
      try {
        const resSearch = await airtableFetch(`Foods?maxRecords=10`);
        if (resSearch && Array.isArray(resSearch.records)) {
          matches = resSearch.records.map(r => ({ id: r.id, content: r.fields['Food Name'] || r.fields['Name'] || '', metadata: r.fields }));
        }
      } catch (e) { /* ignore */ }
    }
  }

  // Return caption + matches to client
  res.status(200).json({
    imageUrl,
    caption,
    matches // array of {id, content, metadata, distance?}
  });
}


// --- Utility: parse JSON body safely (works for Vercel)
async function parseJson(req) {
  try {
    if (req.headers && req.headers['content-type'] && req.headers['content-type'].includes('application/json')) {
      return req.body && Object.keys(req.body).length ? req.body : JSON.parse(await getRawBody(req));
    }
    // otherwise attempt to read
    return req.body ? req.body : JSON.parse(await getRawBody(req));
  } catch (e) {
    return {};
  }
}

// Helper to read raw body if required
function getRawBody(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', chunk => data += chunk);
    req.on('end', () => resolve(data));
    req.on('error', err => reject(err));
  });
}
