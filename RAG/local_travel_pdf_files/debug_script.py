#!/usr/bin/env python3
"""
Smart debug script that auto-detects the travel folder location
"""
import os
import glob
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

def find_travel_folder():
    """Automatically find the travel folder"""
    possible_paths = [
        "./travel",           # Same directory
        "../travel",          # One level up
        "../../travel",       # Two levels up
        "./local_travel_pdf_files/../travel",  # Specific path
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            pdf_files = glob.glob(f"{path}/*.pdf")
            if pdf_files:
                print(f"✅ Found travel folder at: {os.path.abspath(path)}")
                return path
    
    print("❌ Could not find travel folder with PDF files")
    return None

def check_ollama_connection():
    """Test if Ollama is running and model is available"""
    print("=== OLLAMA CONNECTION TEST ===")
    try:
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        test_embedding = embeddings.embed_query("test")
        print(f"✅ Ollama connection successful!")
        print(f"✅ mxbai-embed-large model working!")
        print(f"✅ Embedding dimension: {len(test_embedding)}")
        return True
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("Make sure Ollama is running: ollama serve")
        print("Make sure model is installed: ollama pull mxbai-embed-large")
        return False

def check_pdf_files():
    """Check if PDF files exist and are readable"""
    print("\n=== PDF FILES CHECK ===")
    
    travel_folder = find_travel_folder()
    if not travel_folder:
        return False
    
    pdf_files = glob.glob(f"{travel_folder}/*.pdf")
    print(f"Found {len(pdf_files)} PDF files:")
    
    for pdf in pdf_files:
        file_size = os.path.getsize(pdf) / 1024  # KB
        print(f"  📄 {os.path.basename(pdf)} ({file_size:.1f} KB)")
    
    return len(pdf_files) > 0

def test_pdf_loading():
    """Test loading a single PDF"""
    print("\n=== PDF LOADING TEST ===")
    try:
        from langchain_community.document_loaders import PyPDFLoader
        
        travel_folder = find_travel_folder()
        if not travel_folder:
            print("❌ No travel folder found")
            return False
        
        pdf_files = glob.glob(f"{travel_folder}/*.pdf")
        if not pdf_files:
            print("❌ No PDF files to test")
            return False
        
        test_pdf = pdf_files[0]
        print(f"Testing with: {os.path.basename(test_pdf)}")
        
        loader = PyPDFLoader(test_pdf)
        pages = loader.load_and_split()
        
        print(f"✅ Successfully loaded {len(pages)} pages")
        
        if pages:
            first_page = pages[0]
            print(f"✅ First page content length: {len(first_page.page_content)} characters")
            content_preview = first_page.page_content[:200].replace('\n', ' ')
            print(f"Content preview: {content_preview}...")
            
            # Check if content is meaningful (not just whitespace/garbage)
            if len(first_page.page_content.strip()) < 10:
                print("⚠️  Warning: Page content seems very short or empty")
        
        return True
        
    except Exception as e:
        print(f"❌ PDF loading failed: {e}")
        return False

def check_vector_database():
    """Check existing vector database"""
    print("\n=== VECTOR DATABASE CHECK ===")
    db_location = "./chroma_travel_db"
    
    if not os.path.exists(db_location):
        print(f"❌ No existing database found at {db_location}")
        return False
    
    try:
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        vector_store = Chroma(
            collection_name="travel_documents",
            persist_directory=db_location,
            embedding_function=embeddings
        )
        
        count = vector_store._collection.count()
        print(f"✅ Database loaded successfully!")
        print(f"✅ Document count: {count}")
        
        if count > 0:
            # Test search
            results = vector_store.similarity_search("reservation", k=3)
            print(f"✅ Search test successful! Found {len(results)} results")
            
            for i, doc in enumerate(results):
                print(f"  Result {i+1}: {doc.metadata.get('source', 'Unknown')} (page {doc.metadata.get('page', '?')})")
        else:
            print("⚠️  Database exists but contains no documents")
        
        return True
        
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        return False

def rebuild_database():
    """Offer to rebuild the database"""
    print("\n=== DATABASE REBUILD OPTION ===")
    
    travel_folder = find_travel_folder()
    if not travel_folder:
        print("❌ Cannot rebuild - no travel folder found")
        return False
    
    response = input("Would you like to rebuild the vector database? (y/n): ").lower()
    if response != 'y':
        return False
    
    # Remove existing database
    db_location = "./chroma_travel_db"
    if os.path.exists(db_location):
        import shutil
        shutil.rmtree(db_location)
        print("🗑️  Removed existing database")
    
    print("🔄 Rebuilding database...")
    try:
        # Import and run vector store creation
        exec(open("vector_store_PDF.py").read())
        print("✅ Database rebuilt successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to rebuild database: {e}")
        return False

def main():
    """Run all diagnostic checks"""
    print("🔍 SMART VECTOR STORE DIAGNOSTICS")
    print("=" * 50)
    
    checks = [
        ("Ollama Connection", check_ollama_connection),
        ("PDF Files", check_pdf_files),
        ("PDF Loading", test_pdf_loading),
        ("Vector Database", check_vector_database)
    ]
    
    results = {}
    for name, check_func in checks:
        results[name] = check_func()
    
    print("\n" + "=" * 50)
    print("🏁 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All checks passed! Your setup should work.")
    else:
        print("\n⚠️  Some checks failed.")
        
        # If PDFs load but database is empty, offer to rebuild
        if results.get("PDF Loading") and not results.get("Vector Database"):
            rebuild_database()
    
    return all_passed

if __name__ == "__main__":
    main()