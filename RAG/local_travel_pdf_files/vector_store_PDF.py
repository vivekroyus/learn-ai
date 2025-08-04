from langchain_core.tools import retriever
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
import os
import glob

print("=== VECTOR STORE INITIALIZATION DEBUG ===")

# Initialize embeddings - you can switch between these models:
embeddings = OllamaEmbeddings(model="mxbai-embed-large")      # Current choice - excellent performance
# embeddings = OllamaEmbeddings(model="nomic-embed-text")         # Alternative - optimized for long documents

print(f"Using embedding model: mxbai-embed-large")

# Database location
db_location = "./chroma_travel_db"
add_documents = not os.path.exists(db_location)

print(f"Database location: {db_location}")
print(f"Database exists: {os.path.exists(db_location)}")
print(f"Will add documents: {add_documents}")

# Initialize vector store
print("Initializing Chroma vector store...")
try:
    vector_store = Chroma(
        collection_name="travel_documents",
        persist_directory=db_location,
        embedding_function=embeddings
    )
    print("✅ Vector store initialized successfully!")
except Exception as e:
    print(f"❌ Error initializing vector store: {e}")
    raise

# Check if collection already has documents
try:
    existing_count = vector_store._collection.count()
    print(f"Existing documents in collection: {existing_count}")
except Exception as e:
    print(f"Could not get existing document count: {e}")
    existing_count = 0

if add_documents:
    documents = []
    ids = []
    
    # Auto-detect travel folder location
    possible_paths = ["./travel", "../travel", "../../travel"]
    travel_folder = None
    
    for path in possible_paths:
        if os.path.exists(path):
            test_pdfs = glob.glob(f"{path}/*.pdf")
            if test_pdfs:
                travel_folder = path
                print(f"✅ Found travel folder at: {os.path.abspath(travel_folder)}")
                break
    
    if not travel_folder:
        print("❌ Could not find travel folder with PDF files!")
        print("Checked paths:")
        for path in possible_paths:
            print(f"  - {os.path.abspath(path)}")
        raise FileNotFoundError("Travel folder not found")
    
    pdf_files = glob.glob(f"{travel_folder}/*.pdf")
    print(f"Found {len(pdf_files)} PDF files:")
    
    for pdf in pdf_files:
        print(f"  - {pdf}")
    
    if len(pdf_files) == 0:
        print("❌ No PDF files found! Check your travel folder path.")
    
    total_pages = 0
    successful_files = 0
    
    for pdf_file in pdf_files:
        print(f"\n📄 Processing: {os.path.basename(pdf_file)}")
        try:
            # Load PDF using PyPDFLoader
            loader = PyPDFLoader(pdf_file)
            pages = loader.load_and_split()
            
            print(f"  - Extracted {len(pages)} pages")
            
            # Process each page
            for i, page in enumerate(pages):
                # Extract filename for metadata
                filename = os.path.basename(pdf_file)
                
                # Show first 200 characters of content for verification
                content_preview = page.page_content[:200].replace('\n', ' ')
                print(f"    Page {i+1}: {content_preview}...")
                
                # Create document with content and metadata
                document = Document(
                    page_content=page.page_content,
                    metadata={
                        "source": filename,
                        "page": i + 1,
                        "file_path": pdf_file,
                        "content_length": len(page.page_content)
                    }
                )
                
                # Create unique ID for each page
                doc_id = f"{filename}_page_{i+1}"
                
                ids.append(doc_id)
                documents.append(document)
                total_pages += 1
                
            successful_files += 1
            print(f"  ✅ Successfully processed {len(pages)} pages")
                
        except Exception as e:
            print(f"  ❌ Error processing {pdf_file}: {str(e)}")
            continue
    
    print(f"\n=== PROCESSING SUMMARY ===")
    print(f"Total PDF files found: {len(pdf_files)}")
    print(f"Successfully processed files: {successful_files}")
    print(f"Total pages extracted: {total_pages}")
    print(f"Documents to add to vector store: {len(documents)}")
    
    if documents:
        print(f"\n🔄 Adding {len(documents)} document pages to vector store...")
        try:
            # Test embedding generation first
            print("Testing embedding generation...")
            test_text = documents[0].page_content[:500]  # Test with first 500 chars
            test_embedding = embeddings.embed_query(test_text)
            print(f"✅ Test embedding successful! Dimension: {len(test_embedding)}")
            
            # Add documents to vector store
            vector_store.add_documents(documents=documents, ids=ids)
            print("✅ Documents added successfully to vector store!")
            
            # Verify documents were added
            final_count = vector_store._collection.count()
            print(f"✅ Final document count in collection: {final_count}")
            
        except Exception as e:
            print(f"❌ Error adding documents to vector store: {e}")
            raise
    else:
        print("❌ No documents were processed successfully.")

else:
    print("📁 Using existing vector database")
    try:
        existing_count = vector_store._collection.count()
        print(f"Documents in existing collection: {existing_count}")
        
        # Show sample documents
        if existing_count > 0:
            sample_docs = vector_store.similarity_search("travel", k=2)
            print("\n📋 Sample documents in collection:")
            for i, doc in enumerate(sample_docs):
                print(f"  {i+1}. Source: {doc.metadata.get('source', 'Unknown')}")
                print(f"     Content preview: {doc.page_content[:150]}...")
    except Exception as e:
        print(f"Error accessing existing collection: {e}")

# Create retriever
print("\n🔍 Creating retriever...")
try:
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )
    
    # Test the retriever
    print("Testing retriever with sample query...")
    test_results = retriever.invoke("travel reservation")
    print(f"✅ Retriever test successful! Retrieved {len(test_results)} documents")
    
    if test_results:
        print("Sample retrieved document:")
        sample_doc = test_results[0]
        print(f"  Source: {sample_doc.metadata.get('source', 'Unknown')}")
        print(f"  Page: {sample_doc.metadata.get('page', 'Unknown')}")
        print(f"  Content: {sample_doc.page_content[:200]}...")
    
except Exception as e:
    print(f"❌ Error creating or testing retriever: {e}")
    raise

print("\n✅ Vector store setup complete!")
print("=" * 50)

# Export the retriever for use in other modules
__all__ = ['retriever']