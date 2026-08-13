# OA-RAG Implementation Next Steps

## What I will do first

1. Map the real  schema to contract-risk concepts.
   - Identify which columns should represent the contract title / display name.
   - Identify which columns should provide the searchable contract body text.
   - Identify which columns should represent amount, department, dates, approval flags, and other metadata.
   - Confirm which fields can support contract filtering in the new UI.

2. Read the remaining corpchat source files that drive search and UI behavior.
   - 
   - 
   - 
   - 
   - 
   - 
   - 

3. Implement the migration in this order.
   - Convert configuration and DB helpers to MySQL.
   - Rewrite the search layer to index .
   - Rewrite the Streamlit UI as a contract search / browser / dashboard app.
   - Remove Onyx Chat references entirely.
   - Validate the result with build and search checks.

## Important constraint

The real MySQL table does **not** contain the assumed fields , , , or , so those concepts must be mapped to the actual schema before implementation.
