import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path

def ensure_directories():
    """Créer les répertoires nécessaires s'ils n'existent pas"""
    Path("faiss_index").mkdir(exist_ok=True)
    Path("docs").mkdir(exist_ok=True)

    

def build_index(json_path, index_path, docs_path):
    """Construire l'index FAISS pour du contenu avec exemples de code"""
    print(f"📚 Construction de l'index pour : {json_path}")
    
    # Vérifier si le fichier source existe
    if not os.path.exists(json_path):
        print(f"❌ Fichier source non trouvé : {json_path}")
        return False
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not data:
            print(f"❌ Fichier JSON vide : {json_path}")
            return False
        
        print(f"📄 Chargement de {len(data)} éléments")
        
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        docs = []
        vectors = []
        
        for i, item in enumerate(data):
            try:
                # Gérer les cas où code_examples pourrait être manquant
                code_examples = item.get("code_examples", [])
                code_block = "\n".join(code_examples) if code_examples else ""
                
                # Construire le texte complet
                title = item.get("title", f"Item {i+1}")
                content = item.get("content", "")
                
                full_text = f"{title}\n\n{content}"
                if code_block:
                    full_text += f"\n\n{code_block}"
                
                docs.append(full_text)
                vec = model.encode(full_text)
                vectors.append(vec)
                
            except Exception as e:
                print(f"⚠️ Erreur lors du traitement de l'item {i+1}: {str(e)}")
                continue
        
        if not vectors:
            print("❌ Aucun vecteur généré")
            return False
        
        print(f"🔄 Création de l'index FAISS avec {len(vectors)} vecteurs")
        
        # Créer l'index FAISS
        dim = len(vectors[0])
        index = faiss.IndexFlatL2(dim)
        vectors_array = np.array(vectors).astype("float32")
        index.add(vectors_array)
        
        # Sauvegarder l'index et les documents
        ensure_directories()
        faiss.write_index(index, index_path)
        
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(docs, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Index créé avec succès : {index_path}")
        print(f"✅ Documents sauvegardés : {docs_path}")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la construction de l'index : {str(e)}")
        return False

def build_index_from_slides(json_path, index_path, docs_path):
    """Construire l'index FAISS pour du contenu de slides"""
    print(f"📚 Construction de l'index slides pour : {json_path}")
    
    # Vérifier si le fichier source existe
    if not os.path.exists(json_path):
        print(f"❌ Fichier source non trouvé : {json_path}")
        return False
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not data:
            print(f"❌ Fichier JSON vide : {json_path}")
            return False
        
        print(f"📄 Chargement de {len(data)} slides")
        
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        docs = []
        vectors = []
        
        for slide in data:
            try:
                slide_number = slide.get("slide_number", "Unknown")
                title = slide.get("title", f"Slide {slide_number}")
                content = slide.get("content", "")
                
                # Créer le texte de la slide
                slide_text = f"Slide {slide_number}: {title}\n\n{content}"
                
                docs.append(slide_text)
                vec = model.encode(slide_text)
                vectors.append(vec)
                
            except Exception as e:
                print(f"⚠️ Erreur lors du traitement de la slide {slide.get('slide_number', 'Unknown')}: {str(e)}")
                continue
        
        if not vectors:
            print("❌ Aucun vecteur généré")
            return False
        
        print(f"🔄 Création de l'index FAISS avec {len(vectors)} vecteurs")
        
        # Créer l'index FAISS
        dim = len(vectors[0])
        index = faiss.IndexFlatL2(dim)
        vectors_array = np.array(vectors).astype("float32")
        index.add(vectors_array)
        
        # Sauvegarder l'index et les documents
        ensure_directories()
        faiss.write_index(index, index_path)
        
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(docs, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Index slides créé avec succès : {index_path}")
        print(f"✅ Documents sauvegardés : {docs_path}")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la construction de l'index slides : {str(e)}")
        return False

def main():
    """Fonction principale pour créer tous les index"""
    print("🚀 Démarrage de la construction des index FAISS\n")
    
    # Configuration des domaines
    domains = [
        {
            "name": "Angular",
            "source": "RAG_Content/angular_training.json",
            "index": "faiss_index/angular_faiss.index",
            "docs": "docs/angular_docs.json",
            "type": "standard"
        },
        {
            "name": "Java",
            "source": "RAG_Content/java_training.json", 
            "index": "faiss_index/java_faiss.index",
            "docs": "docs/java_docs.json",
            "type": "standard"
        },
        {
            "name": "Spring JEE",
            "source": "RAG_Content/spring_jee.json",
            "index": "faiss_index/spring_jee_faiss.index", 
            "docs": "docs/spring_jee_slides_docs.json",
            "type": "slides"
        }
    ]
    
    results = []
    
    for domain in domains:
        print(f"\n{'='*50}")
        print(f"🏗️ Construction de l'index pour {domain['name']}")
        print(f"{'='*50}")
        
        if domain["type"] == "slides":
            success = build_index_from_slides(domain["source"], domain["index"], domain["docs"])
        else:
            success = build_index(domain["source"], domain["index"], domain["docs"])
        
        results.append({
            "domain": domain["name"],
            "success": success
        })
    
    # Résumé final
    print(f"\n{'='*50}")
    print("📊 RÉSUMÉ DE LA CONSTRUCTION")
    print(f"{'='*50}")
    
    for result in results:
        status = "✅ SUCCÈS" if result["success"] else "❌ ÉCHEC"
        print(f"{result['domain']}: {status}")
    
    successful = sum(1 for r in results if r["success"])
    total = len(results)
    print(f"\n🎯 {successful}/{total} index créés avec succès")
    
    if successful == total:
        print("🎉 Tous les index ont été créés avec succès!")
    else:
        print("⚠️ Certains index n'ont pas pu être créés. Vérifiez les messages d'erreur ci-dessus.")

if __name__ == "__main__":
    main()