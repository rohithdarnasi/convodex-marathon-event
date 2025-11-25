from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import google.generativeai as genai
from google.cloud import firestore
import hashlib
import time

app = Flask(__name__)
CORS(app)

# Initialize Gemini
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Firestore
db = firestore.Client()

# Configuration
MAX_PAGES = 50  # Limit pages to crawl for the 1-hour demo
CHUNK_SIZE = 800  # Characters per chunk

def crawl_website(start_url):
    """Crawls a website and extracts text content from all pages"""
    visited = set()
    to_visit = [start_url]
    all_content = []
    base_domain = urlparse(start_url).netloc
    
    print(f"Starting crawl of {start_url}")
    
    while to_visit and len(visited) < MAX_PAGES:
        url = to_visit.pop(0)
        
        if url in visited:
            continue
            
        try:
            print(f"Crawling: {url}")
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            if response.status_code != 200:
                continue
                
            visited.add(url)
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            if len(text) > 100:  # Only save pages with substantial content
                all_content.append({
                    'url': url,
                    'title': soup.title.string if soup.title else url,
                    'content': text
                })
            
            # Find more links on the same domain
            for link in soup.find_all('a', href=True):
                full_url = urljoin(url, link['href'])
                
                # Only crawl links on the same domain
                if urlparse(full_url).netloc == base_domain:
                    # Remove fragments and query params
                    clean_url = full_url.split('#')[0].split('?')[0]
                    
                    if clean_url not in visited and clean_url not in to_visit:
                        to_visit.append(clean_url)
        
        except Exception as e:
            print(f"Error crawling {url}: {str(e)}")
            continue
    
    print(f"Crawled {len(visited)} pages, extracted {len(all_content)} pages with content")
    return all_content

def chunk_content(content_list):
    """Split content into smaller chunks"""
    chunks = []
    
    for page in content_list:
        text = page['content']
        # Split into chunks
        for i in range(0, len(text), CHUNK_SIZE):
            chunk_text = text[i:i+CHUNK_SIZE]
            if len(chunk_text) > 100:  # Only keep meaningful chunks
                chunks.append({
                    'url': page['url'],
                    'title': page['title'],
                    'content': chunk_text,
                    'chunk_index': i // CHUNK_SIZE
                })
    
    print(f"Created {len(chunks)} chunks")
    return chunks

def generate_embeddings(chunks, chatbot_id):
    """Generate embeddings and store in Firestore"""
    print("Generating embeddings...")
    
    for idx, chunk in enumerate(chunks):
        try:
            # Generate embedding using Gemini
            result = genai.embed_content(
                model="models/embedding-001",
                content=chunk['content'],
                task_type="retrieval_document"
            )
            
            embedding = result['embedding']
            
            # Store in Firestore
            doc_ref = db.collection('chatbots').document(chatbot_id).collection('chunks').document()
            doc_ref.set({
                'url': chunk['url'],
                'title': chunk['title'],
                'content': chunk['content'],
                'chunk_index': chunk['chunk_index'],
                'embedding': embedding,
                'created_at': firestore.SERVER_TIMESTAMP
            })
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(chunks)} chunks")
                
        except Exception as e:
            print(f"Error generating embedding for chunk {idx}: {str(e)}")
            continue
    
    print("Embeddings generated and stored")

def find_relevant_chunks(chatbot_id, query, top_k=3):
    """Find most relevant chunks for a query using cosine similarity"""
    # Generate query embedding
    result = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )
    query_embedding = result['embedding']
    
    # Get all chunks
    chunks_ref = db.collection('chatbots').document(chatbot_id).collection('chunks')
    chunks = chunks_ref.stream()
    
    # Calculate similarity
    similarities = []
    for chunk in chunks:
        chunk_data = chunk.to_dict()
        chunk_embedding = chunk_data['embedding']
        
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(query_embedding, chunk_embedding))
        magnitude1 = sum(a * a for a in query_embedding) ** 0.5
        magnitude2 = sum(b * b for b in chunk_embedding) ** 0.5
        similarity = dot_product / (magnitude1 * magnitude2)
        
        similarities.append({
            'content': chunk_data['content'],
            'url': chunk_data['url'],
            'title': chunk_data['title'],
            'similarity': similarity
        })
    
    # Sort by similarity
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    return similarities[:top_k]

@app.route('/api/create-chatbot', methods=['POST'])
def create_chatbot():
    """Main endpoint to create a chatbot from a website URL"""
    try:
        data = request.json
        website_url = data.get('url')
        
        if not website_url:
            return jsonify({'error': 'URL is required'}), 400
        
        # Generate unique chatbot ID
        chatbot_id = hashlib.md5(f"{website_url}{time.time()}".encode()).hexdigest()[:12]
        
        # Step 1: Crawl website
        print(f"Step 1: Crawling {website_url}")
        content = crawl_website(website_url)
        
        if not content:
            return jsonify({'error': 'No content found on the website'}), 400
        
        # Step 2: Chunk content
        print("Step 2: Chunking content")
        chunks = chunk_content(content)
        
        # Step 3: Generate embeddings and store
        print("Step 3: Generating embeddings")
        generate_embeddings(chunks, chatbot_id)
        
        # Step 4: Store chatbot metadata
        db.collection('chatbots').document(chatbot_id).set({
            'url': website_url,
            'created_at': firestore.SERVER_TIMESTAMP,
            'num_chunks': len(chunks),
            'num_pages': len(content)
        })
        
        print(f"Chatbot created successfully: {chatbot_id}")
        
        return jsonify({
            'success': True,
            'chatbot_id': chatbot_id,
            'pages_crawled': len(content),
            'chunks_created': len(chunks)
        })
        
    except Exception as e:
        print(f"Error creating chatbot: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Endpoint for chatbot to answer questions"""
    try:
        data = request.json
        chatbot_id = data.get('chatbot_id')
        message = data.get('message')
        
        if not chatbot_id or not message:
            return jsonify({'error': 'chatbot_id and message are required'}), 400
        
        # Find relevant chunks
        relevant_chunks = find_relevant_chunks(chatbot_id, message)
        
        if not relevant_chunks:
            return jsonify({
                'response': "I couldn't find any relevant information to answer your question."
            })
        
        # Build context from relevant chunks
        context = "\n\n".join([
            f"From {chunk['title']} ({chunk['url']}):\n{chunk['content']}"
            for chunk in relevant_chunks
        ])
        
        # Generate response using Gemini
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        prompt = f"""You are a helpful chatbot assistant. Answer the user's question based ONLY on the following context from the website. If the answer is not in the context, say "I don't have information about that in my knowledge base."

Context:
{context}

User Question: {message}

Answer:"""
        
        response = model.generate_content(prompt)
        
        return jsonify({
            'response': response.text,
            'sources': [{'title': chunk['title'], 'url': chunk['url']} for chunk in relevant_chunks]
        })
        
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
