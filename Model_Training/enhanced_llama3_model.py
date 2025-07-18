# enhanced_llama3_model.py
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import subprocess
import os
import time
import re
import requests
import base64
from typing import Optional, List, Dict

# Configuration des APIs d'images
IMAGE_APIS = {
    "stability": {
        "url": "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
        "key": ""
    },
    "dall_e": {
        "url": "https://api.openai.com/v1/images/generations",
        "key": ""
    },
    "local_sd": {
        "url": "http://localhost:7860/sdapi/v1/txt2img",  # Automatic1111
        "key": None
    }
}

# Prompts pour générer des descriptions d'images
IMAGE_DESCRIPTION_PROMPTS = {
    "fr": """En plus du contenu précédent, génère une description d'image technique qui serait utile pour illustrer "{current_topic}".

FORMAT:
**Description d'image** : Une description précise en 1-2 phrases d'un diagramme, schéma ou illustration qui aiderait à comprendre {current_topic}. Sois spécifique sur les éléments visuels (boîtes, flèches, couleurs, disposition).

Exemple: "Diagramme montrant l'architecture MVC avec 3 boîtes colorées (Model en bleu, View en vert, Controller en orange) reliées par des flèches bidirectionnelles sur fond blanc."
""",
    "en": """In addition to the previous content, generate a technical image description that would be useful to illustrate "{current_topic}".

FORMAT:
**Image description** : A precise description in 1-2 sentences of a diagram, schema or illustration that would help understand {current_topic}. Be specific about visual elements (boxes, arrows, colors, layout).

Example: "Diagram showing MVC architecture with 3 colored boxes (Model in blue, View in green, Controller in orange) connected by bidirectional arrows on white background."
"""
}

def generate_image_description(content: str, current_topic: str, sujet: str, lang: str = "fr") -> str:
    """Génère une description d'image à partir du contenu"""
    
    # Ajouter le prompt pour la description d'image
    image_prompt = IMAGE_DESCRIPTION_PROMPTS.get(lang, IMAGE_DESCRIPTION_PROMPTS["fr"])
    enhanced_prompt = content + "\n\n" + image_prompt.format(current_topic=current_topic, sujet=sujet)
    
    try:
        response = generate_response(enhanced_prompt, max_retries=2, timeout=120)
        
        # Extraire la description d'image
        patterns = [
            r'\*\*Description d\'image\*\*\s*:?\s*(.+?)(?=\n\n|\*\*|$)',
            r'\*\*Image description\*\*\s*:?\s*(.+?)(?=\n\n|\*\*|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return f"Technical diagram illustrating {current_topic} in {sujet}"
        
    except Exception as e:
        print(f"❌ Erreur génération description image: {e}")
        return f"Technical diagram for {current_topic}"

def generate_image_with_stability(description: str, output_path: str) -> bool:
    """Génère une image avec Stability AI"""
    try:
        headers = {
            "Authorization": f"Bearer {IMAGE_APIS['stability']['key']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "text_prompts": [
                {
                    "text": f"Technical diagram, {description}, clean minimal design, white background, professional, educational",
                    "weight": 1
                }
            ],
            "cfg_scale": 7,
            "height": 512,
            "width": 512,
            "samples": 1,
            "steps": 30,
            "style_preset": "digital-art"
        }
        
        response = requests.post(IMAGE_APIS['stability']['url'], headers=headers, json=data)
        
        if response.status_code == 200:
            data = response.json()
            image_data = base64.b64decode(data["artifacts"][0]["base64"])
            
            with open(output_path, "wb") as f:
                f.write(image_data)
            
            return True
        else:
            print(f"❌ Erreur Stability AI: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur génération image: {e}")
        return False

def generate_image_with_local_sd(description: str, output_path: str) -> bool:
    """Génère une image avec Stable Diffusion local (Automatic1111)"""
    try:
        data = {
            "prompt": f"Technical diagram, {description}, clean minimal design, white background, professional, educational, software architecture",
            "negative_prompt": "blurry, low quality, text, watermark, signature, people, faces",
            "steps": 20,
            "width": 512,
            "height": 512,
            "cfg_scale": 7,
            "sampler_name": "Euler a"
        }
        
        response = requests.post(IMAGE_APIS['local_sd']['url'], json=data)
        
        if response.status_code == 200:
            result = response.json()
            image_data = base64.b64decode(result["images"][0])
            
            with open(output_path, "wb") as f:
                f.write(image_data)
            
            return True
        else:
            print(f"❌ Erreur Stable Diffusion local: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur génération image locale: {e}")
        return False

def create_mermaid_diagram(description: str, topic: str, subject: str) -> str:
    """Crée un diagramme Mermaid basé sur la description"""
    
    # Templates Mermaid selon le sujet
    templates = {
        "java": """
graph TD
    A[Java Application] --> B[Main Class]
    B --> C[Methods]
    C --> D[Variables]
    D --> E[Output]
""",
        "angular": """
graph TD
    A[Angular App] --> B[Components]
    B --> C[Services]
    C --> D[Models]
    A --> E[Templates]
    A --> F[Routing]
""",
        "jee": """
graph TD
    A[Client] --> B[Controller]
    B --> C[Service]
    C --> D[Repository]
    D --> E[Database]
    B --> F[View]
"""
    }
    
    base_template = templates.get(subject, templates["java"])
    
    # Personnaliser selon le topic
    if "mvc" in topic.lower():
        return """
graph TD
    A[Model] --> B[View]
    B --> C[Controller]
    C --> A
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#fff3e0
"""
    elif "architecture" in topic.lower():
        return """
graph TD
    A[Presentation Layer] --> B[Business Layer]
    B --> C[Data Layer]
    C --> D[Database]
    style A fill:#e8f5e8
    style B fill:#fff3e0
    style C fill:#e1f5fe
    style D fill:#fce4ec
"""
    
    return base_template

def generate_visual_content(description: str, topic: str, subject: str, slide_number: int) -> Dict:
    """Génère du contenu visuel (image + diagramme)"""
    
    # Créer les dossiers nécessaires
    os.makedirs("images", exist_ok=True)
    os.makedirs("diagrams", exist_ok=True)
    
    visual_content = {
        "image_path": None,
        "mermaid_diagram": None,
        "description": description
    }
    
    # 1. Tenter de générer une image
    image_path = f"images/slide_{slide_number}_{subject}_{topic.replace(' ', '_')}.png"
    
    # Essayer d'abord Stable Diffusion local
    if generate_image_with_local_sd(description, image_path):
        visual_content["image_path"] = image_path
        print(f"✅ Image générée: {image_path}")
    # Sinon essayer Stability AI (si configuré)
    elif IMAGE_APIS['stability']['key'] != "YOUR_STABILITY_API_KEY":
        if generate_image_with_stability(description, image_path):
            visual_content["image_path"] = image_path
            print(f"✅ Image générée: {image_path}")
    
    # 2. Générer un diagramme Mermaid
    mermaid_diagram = create_mermaid_diagram(description, topic, subject)
    visual_content["mermaid_diagram"] = mermaid_diagram
    
    # Sauvegarder le diagramme
    diagram_path = f"diagrams/slide_{slide_number}_{subject}_{topic.replace(' ', '_')}.mmd"
    with open(diagram_path, "w", encoding="utf-8") as f:
        f.write(mermaid_diagram)
    
    print(f"✅ Diagramme Mermaid créé: {diagram_path}")
    
    return visual_content

# Modifier le SYSTEM_PROMPT pour inclure les images
ENHANCED_SYSTEM_PROMPT = {
    "fr": """Tu es un assistant pédagogique expert. Génère une formation {niveau} sur '{sujet}'.

SLIDE {slide_number}: "{current_topic}"

FORMAT OBLIGATOIRE:
1. **Explication orale** (4-5 phrases max, concise et claire)
2. **Résumé HTML** - UTILISE UNIQUEMENT DES BALISES HTML VALIDES:
   <ul>
   <li><strong>Point 1</strong>: Description courte</li>
   <li><strong>Point 2</strong>: Description courte</li>
   <li><strong>Point 3</strong>: Description courte</li>
   </ul>
3. **Exemples de code** (si nécessaire) au FORMAT HTML avec balises <pre><code>
4. **Description d'image** : Description précise d'un diagramme technique qui illustrerait {current_topic}

CONTEXTE: {context_summary}

IMPORTANT: 
- Sois spécifique sur les éléments visuels (boîtes, flèches, couleurs)
- Termine par "Souhaitez-vous continuer ?"
- RÉPONDS EN FRANÇAIS UNIQUEMENT.
""",
    "en": """You are an expert teaching assistant. Generate a {niveau} training on '{sujet}'.

SLIDE {slide_number}: "{current_topic}"

MANDATORY FORMAT:
1. **Spoken explanation** (4-5 sentences max, concise and clear)
2. **HTML summary** - USE ONLY VALID HTML TAGS:
   <ul>
   <li><strong>Point 1</strong>: Short description</li>
   <li><strong>Point 2</strong>: Short description</li>
   <li><strong>Point 3</strong>: Short description</li>
   </ul>
3. **Code examples** (if necessary) in HTML FORMAT with <pre><code> tags
4. **Image description** : Precise description of a technical diagram that would illustrate {current_topic}

CONTEXT: {context_summary}

IMPORTANT: 
- Be specific about visual elements (boxes, arrows, colors)
- End with "Would you like to continue?"
- RESPOND IN ENGLISH ONLY.
"""
}

def enhanced_rag_query(query: str, sujet: str, niveau: str, plan: str, history: str, current_topic: str, slide_number: int, lang: str = "fr", top_k: int = 3) -> Dict:
    """RAG query améliorée avec génération de contenu visuel"""
    
    # Génération du contenu textuel (code existant)
    textual_response = rag_query(query, sujet, niveau, plan, history, current_topic, slide_number, lang, top_k)
    
    # Génération de la description d'image
    image_description = generate_image_description(textual_response, current_topic, sujet, lang)
    
    # Génération du contenu visuel
    visual_content = generate_visual_content(image_description, current_topic, sujet, slide_number)
    
    return {
        "textual_content": textual_response,
        "visual_content": visual_content,
        "slide_number": slide_number,
        "topic": current_topic
    }

def save_enhanced_slides_to_json(slides, filename):
    """Sauvegarde les slides avec contenu visuel"""
    data = {
        "slides": slides,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_slides": len(slides)
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Slides enrichies sauvegardées dans {filename}")

# Exemple d'utilisation dans la fonction main
def enhanced_main():
    print("\n📚 Formation interactive avec illustrations visuelles")
    print("🎨 Génération automatique d'images et diagrammes")
    print("Langues : [fr]ançais, [en]glish")
    print("Domaines disponibles :", ", ".join(AVAILABLE_DOMAINS.keys()))
    
    # Configuration (même logique que l'original)
    lang = input("🌐 Langue [fr/en] : ").strip().lower()
    if lang not in ["fr", "en"]:
        lang = "fr"
    
    sujet = input("📌 Sujet (ex: angular, java, jee) : ").strip().lower()
    niveau = input("🎓 Niveau (débutant / intermédiaire / avancé) : ").strip().lower()
    
    # Vérifier la disponibilité des outils de génération d'images
    print("\n🔍 Vérification des outils de génération d'images...")
    
    # Test Stable Diffusion local
    try:
        test_response = requests.get("http://localhost:7860/", timeout=5)
        print("✅ Stable Diffusion local détecté")
    except:
        print("❌ Stable Diffusion local non disponible")
    
    # Logique de génération des slides (adaptée)
    enhanced_slides = []
    
    # ... (même logique que l'original mais avec enhanced_rag_query)
    
    print("\n✅ Formation enrichie générée avec succès!")
    print("📁 Fichiers créés:")
    print("   - Images dans ./images/")
    print("   - Diagrammes dans ./diagrams/")
    print("   - Slides JSON enrichies")

if __name__ == "__main__":
    enhanced_main()