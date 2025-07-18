import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import subprocess
import os
import time
import re
from typing import Optional

# ==== Domaines disponibles ====
AVAILABLE_DOMAINS = {
    "angular": {"index": "faiss_index/angular_faiss.index", "docs": "docs/angular_docs.json"},
    "java": {"index": "faiss_index/java_faiss.index", "docs": "docs/java_docs.json"},
    "jee": {
        "index": "faiss_index/spring_jee_faiss.index",
        "docs": "docs/spring_jee_slides_docs.json"
    }
}

# ==== Modèle d'embedding ====
model = SentenceTransformer("all-MiniLM-L6-v2")

# ==== Prompt multilingue avec instructions HTML STRICTES et exemples de code ====
SYSTEM_PROMPT = {
    "fr": """Tu es un assistant pédagogique expert. Génère une formation {niveau} sur '{sujet}'.

SLIDE {slide_number}: "{current_topic}"

FORMAT OBLIGATOIRE:
1. **Explication orale** (aussi longue que nécessaire, mais toujours claire, structurée et pédagogique)
   - Explique en détail et de manière pédagogique chacun des trois points qui seront listés dans le résumé HTML.
   - Si un exemple de code est fourni, explique également oralement ce que fait le code, comment il fonctionne, et en quoi il illustre les points du résumé.
2. **Résumé HTML** - UTILISE UNIQUEMENT DES BALISES HTML VALIDE:
   <ul>
   <li><strong>Point 1</strong>: Description courte</li>
   <li><strong>Point 2</strong>: Description courte</li>
   <li><strong>Point 3</strong>: Description courte</li>
   </ul>
3. **Exemples de code** (si nécessaire pour le sujet) au FORMAT HTML avec balises <pre><code> :
   <pre><code class='language-{sujet}'>
   // Code d'exemple pertinent et commenté
   </code></pre>

CONTEXTE: {context_summary}

IMPORTANT:
- Le résumé HTML doit être du HTML valide uniquement, sans texte brut.
- Les trois points du résumé HTML doivent impérativement être expliqués dans la partie Explication orale.
- Ajoute des exemples de code pratiques et bien commentés quand c'est pertinent.
- L'explication orale doit également commenter le code fourni en expliquant son fonctionnement.
- Termine toujours par la question : "Souhaitez-vous continuer ?"
- RÉPONDS EN FRANÇAIS UNIQUEMENT.
""",
    
# Les autres langues suivent la même logique :
    "en": """You are an expert teaching assistant. Generate a {niveau} training on '{sujet}'.

SLIDE {slide_number}: "{current_topic}"

MANDATORY FORMAT:
1. **Spoken explanation** (as long as necessary, but always clear, structured, and pedagogical)
   - Explain in detail and pedagogically each of the three points that will appear in the HTML summary.
   - If a code example is provided, orally describe what the code does, how it works, and how it illustrates the summary points.
2. **HTML summary** - USE ONLY VALID HTML TAGS:
   <ul>
   <li><strong>Point 1</strong>: Short description</li>
   <li><strong>Point 2</strong>: Short description</li>
   <li><strong>Point 3</strong>: Short description</li>
   </ul>
3. **Code examples** (if necessary for the topic) in HTML FORMAT with <pre><code> tags:
   <pre><code class='language-{sujet}'>
   // Relevant and commented code example
   </code></pre>

CONTEXT: {context_summary}

IMPORTANT:
- HTML summary must be valid HTML only, not plain text.
- The three summary points MUST be fully explained in the spoken explanation.
- Add practical, well-commented code examples when relevant.
- The spoken explanation must also clarify the purpose and function of the code provided.
- Always end with the question: "Would you like to continue?"
- RESPOND IN ENGLISH ONLY.
""",

# Pareil pour espagnol et italien :
    "es": """Eres un asistente pedagógico experto. Genera una formación {niveau} sobre '{sujet}'.

SLIDE {slide_number}: "{current_topic}"

FORMATO OBLIGATORIO:
1. **Explicación oral** (tan extensa como sea necesario, pero siempre clara, estructurada y pedagógica)
   - Explica en detalle y pedagógicamente cada uno de los tres puntos que aparecerán en el resumen HTML.
   - Si se proporciona un ejemplo de código, describe oralmente qué hace el código, cómo funciona y cómo ejemplifica los puntos del resumen.
2. **Resumen HTML** - USA SOLO ETIQUETAS HTML VÁLIDAS:
   <ul>
   <li><strong>Punto 1</strong>: Descripción corta</li>
   <li><strong>Punto 2</strong>: Descripción corta</li>
   <li><strong>Punto 3</strong>: Descripción corta</li>
   </ul>
3. **Ejemplos de código** (si es necesario para el tema) en FORMATO HTML con etiquetas <pre><code>:
   <pre><code class='language-{sujet}'>
   // Ejemplo de código relevante y comentado
   </code></pre>

CONTEXTO: {context_summary}

IMPORTANTE:
- El resumen HTML debe ser solo HTML válido, no texto plano.
- Los tres puntos del resumen HTML DEBEN explicarse completamente en la explicación oral.
- Añade ejemplos de código prácticos y comentados cuando sea relevante.
- La explicación oral también debe explicar claramente el ejemplo de código proporcionado.
- Termina siempre con la pregunta: "¿Quieres continuar?"
- RESPONDE SOLO EN ESPAÑOL.
""",

    "it": """Sei un assistente pedagogico esperto. Genera una formazione di livello {niveau} su '{sujet}'.

SLIDE {slide_number}: "{current_topic}"

FORMATO OBBLIGATORIO:
1. **Spiegazione orale** (quanto necessario, ma sempre chiara, strutturata e pedagogica)
   - Spiega in dettaglio e con approccio pedagogico ciascuno dei tre punti che appariranno nel riepilogo HTML.
   - Se viene fornito un esempio di codice, descrivi oralmente cosa fa il codice, come funziona e come illustra i punti del riepilogo.
2. **Riepilogo HTML** - USA SOLO TAG HTML VALIDI:
   <ul>
   <li><strong>Punto 1</strong>: Breve descrizione</li>
   <li><strong>Punto 2</strong>: Breve descrizione</li>
   <li><strong>Punto 3</strong>: Breve descrizione</li>
   </ul>
3. **Esempi di codice** (se necessario per l'argomento) in FORMATO HTML con tag <pre><code>:
   <pre><code class='language-{sujet}'>
   // Esempio di codice rilevante e commentato
   </code></pre>

CONTESTO: {context_summary}

IMPORTANTE:
- Il riepilogo HTML deve essere solo HTML valido, non testo semplice.
- I tre punti del riepilogo HTML DEVONO essere spiegati in modo dettagliato nella spiegazione orale.
- Aggiungi esempi di codice pratici e commentati quando è pertinente.
- La spiegazione orale deve anche chiarire il funzionamento dell'esempio di codice fornito.
- Termina sempre con la domanda: "Vuoi continuare?"
- RISPONDI SOLO IN ITALIANO.
"""
}



def convert_to_html_list(text: str) -> str:
    """Convertit du texte en liste HTML si ce n'est pas déjà fait"""
    
    # Si c'est déjà du HTML valide, on retourne tel quel
    if '<ul>' in text and '<li>' in text:
        return text
    
    # Patterns pour détecter les listes
    patterns = [
        r'[•*-]\s*\*\*(.*?)\*\*:?\s*(.*?)(?=\n[•*-]|\n\n|$)',  # • **Point**: Description
        r'[•*-]\s*(.*?)(?=\n[•*-]|\n\n|$)',  # • Point simple
        r'^\d+\.\s*\*\*(.*?)\*\*:?\s*(.*?)(?=\n\d+\.|\n\n|$)',  # 1. **Point**: Description
        r'^\d+\.\s*(.*?)(?=\n\d+\.|\n\n|$)',  # 1. Point simple
    ]
    
    html_items = []
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
        if matches:
            for match in matches:
                if isinstance(match, tuple) and len(match) == 2:
                    title, desc = match
                    html_items.append(f"<li><strong>{title.strip()}</strong>: {desc.strip()}</li>")
                else:
                    html_items.append(f"<li>{match.strip()}</li>")
            break
    
    # Si aucun pattern trouvé, diviser par lignes
    if not html_items:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines[:4]:  # Maximum 4 points
            # Nettoyer la ligne
            clean_line = re.sub(r'^[•*-]\s*', '', line)
            clean_line = re.sub(r'^\d+\.\s*', '', clean_line)
            if clean_line:
                html_items.append(f"<li>{clean_line}</li>")
    
    if html_items:
        return f"<ul>\n{chr(10).join(html_items)}\n</ul>"
    else:
        return "<ul>\n<li>Point clé à retenir</li>\n</ul>"

def post_process_response(response: str) -> str:
    """Post-traite la réponse pour garantir le format HTML"""
    
    # Chercher la section "HTML summary"
    html_section_patterns = [
        r'\*\*HTML [Ss]ummary\*\*:?\s*(.*?)(?=\n\n|\*\*|Would you like|Souhaitez-vous|¿Quieres|```|$)',
        r'\*\*Résumé HTML\*\*:?\s*(.*?)(?=\n\n|\*\*|Would you like|Souhaitez-vous|¿Quieres|```|$)',
        r'\*\*Resumen HTML\*\*:?\s*(.*?)(?=\n\n|\*\*|Would you like|Souhaitez-vous|¿Quieres|```|$)',
    ]
    
    for pattern in html_section_patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            html_content = match.group(1).strip()
            # Convertir en HTML si nécessaire
            html_formatted = convert_to_html_list(html_content)
            # Remplacer dans la réponse
            response = response.replace(match.group(0), f"**HTML summary**:\n{html_formatted}")
            break
    
    return response

def check_ollama_status():
    """Vérifier si Ollama est en cours d'exécution"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10
        )
        return result.returncode == 0
    except:
        return False

def generate_response(prompt: str, max_retries: int = 3, timeout: int = 300) -> str:
    """Génération avec Ollama avec gestion des timeouts et retry"""
    print(f"\n⏳ Génération avec Ollama (timeout: {timeout}s)...")
    
    if not check_ollama_status():
        return "❌ Ollama n'est pas en cours d'exécution. Démarrez-le avec 'ollama serve'"
    
    for attempt in range(max_retries):
        try:
            print(f"🔄 Tentative {attempt + 1}/{max_retries}")
            
            # Utiliser un processus avec timeout plus long
            process = subprocess.Popen(
                ["ollama", "run", "llama3"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            
            # Envoyer le prompt et attendre la réponse
            stdout, stderr = process.communicate(input=prompt, timeout=timeout)
            
            if process.returncode == 0 and stdout.strip():
                print("✅ Génération réussie!")
                # Post-traiter la réponse pour garantir le HTML
                return post_process_response(stdout.strip())
            else:
                print(f"⚠️ Erreur dans la réponse: {stderr}")
                if attempt < max_retries - 1:
                    print(f"🔄 Nouvelle tentative dans 5 secondes...")
                    time.sleep(5)
                    
        except subprocess.TimeoutExpired:
            print(f"❌ Timeout ({timeout}s) - Tentative {attempt + 1}")
            process.kill()
            if attempt < max_retries - 1:
                print("🔄 Nouvelle tentative avec timeout plus long...")
                timeout += 60  # Augmenter le timeout
                time.sleep(5)
        except Exception as e:
            print(f"❌ Erreur: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(5)
    
    return "❌ Échec de la génération après plusieurs tentatives"

def build_optimized_prompt(context: str, sujet: str, niveau: str, current_topic: str, slide_number: int, lang: str = "fr") -> str:
    """Construire un prompt optimisé et plus court"""
    
    # Résumer le contexte si trop long
    context_summary = context[:500] + "..." if len(context) > 500 else context
    if not context_summary.strip():
        context_summary = f"Connaissances générales sur {sujet}"
    
    system_instruction = SYSTEM_PROMPT[lang].format(
        sujet=sujet,
        niveau=niveau,
        current_topic=current_topic,
        slide_number=slide_number,
        context_summary=context_summary
    )
    
    return system_instruction

def rag_query(query: str, sujet: str, niveau: str, plan: str, history: str, current_topic: str, slide_number: int, lang: str = "fr", top_k: int = 3) -> str:
    """RAG query avec gestion d'erreur améliorée"""
    
    if sujet not in AVAILABLE_DOMAINS:
        return f"❌ Domaine '{sujet}' non disponible. Disponibles : {', '.join(AVAILABLE_DOMAINS)}"

    index_path = AVAILABLE_DOMAINS[sujet]["index"]
    docs_path = AVAILABLE_DOMAINS[sujet]["docs"]

    # Vérifier l'existence des fichiers
    if not os.path.exists(index_path):
        return f"❌ Index FAISS manquant: {index_path}. Exécutez build_faiss_index.py"
    
    if not os.path.exists(docs_path):
        return f"❌ Documents manquants: {docs_path}. Exécutez build_faiss_index.py"

    try:
        # Charger l'index et les documents
        index = faiss.read_index(index_path)
        with open(docs_path, "r", encoding="utf-8") as f:
            docs = json.load(f)

        # Requête simplifiée
        enhanced_query = f"{current_topic} {sujet}"
        print(f"🔍 Recherche pour: {enhanced_query}")
        
        # Recherche vectorielle
        query_vector = model.encode([enhanced_query])
        distances, indices = index.search(np.array(query_vector).astype("float32"), top_k)
        
        # Récupérer les documents pertinents
        relevant_docs = []
        for i, idx in enumerate(indices[0]):
            if idx < len(docs) and distances[0][i] < 2.0:  # Seuil de pertinence élargi
                relevant_docs.append(docs[idx][:400])  # Limiter la taille de chaque doc
        
        # Construire le contexte
        context = "\n---\n".join(relevant_docs) if relevant_docs else f"Utilise tes connaissances générales sur {sujet}"
        
        # Construire le prompt optimisé
        prompt = build_optimized_prompt(context, sujet, niveau, current_topic, slide_number, lang)
        
        # Générer la réponse
        return generate_response(prompt, max_retries=2, timeout=180)
        
    except Exception as e:
        error_msg = f"❌ Erreur RAG: {str(e)}"
        print(error_msg)
        
        # Fallback : générer sans contexte
        print("🔄 Génération sans contexte RAG...")
        fallback_prompt = build_optimized_prompt("", sujet, niveau, current_topic, slide_number, lang)
        return generate_response(fallback_prompt, max_retries=1, timeout=120)

def generate_fallback_slide(current_topic: str, slide_number: int, sujet: str, niveau: str, lang: str = "fr") -> str:
    """Génération de slide de secours sans LLM"""
    
    # Générer un exemple de code basique selon le sujet
    code_examples = {
        "java": f"""```java
// Exemple basique pour {current_topic}
public class {current_topic.replace(' ', '')}Example {{
    public static void main(String[] args) {{
        System.out.println("Apprentissage de {current_topic}");
        // Votre code ici
    }}
}}
```""",
        "angular": f"""```typescript
// Exemple basique pour {current_topic}
import {{ Component }} from '@angular/core';

@Component({{
  selector: 'app-{current_topic.lower().replace(' ', '-')}',
  template: '<h1>{current_topic}</h1>'
}})
export class {current_topic.replace(' ', '')}Component {{
  // Votre code ici
}}
```""",
        "jee": f"""```java
// Exemple basique pour {current_topic}
@Entity
public class {current_topic.replace(' ', '')}Entity {{
    @Id
    private Long id;
    // Votre code ici
}}
```"""
    }
    
    templates = {
        "fr": f"""**Explication orale** :
Dans cette partie sur {current_topic}, nous allons explorer les concepts fondamentaux de ce sujet dans le contexte de {sujet}. Cette section est conçue pour un niveau {niveau} et vous donnera les bases nécessaires pour comprendre et appliquer ces concepts. Nous verrons les éléments clés et les bonnes pratiques à retenir.

**Résumé HTML** :
<ul>
<li><strong>Objectif</strong>: Comprendre {current_topic}</li>
<li><strong>Niveau</strong>: {niveau}</li>
<li><strong>Points clés</strong>: Concepts fondamentaux de {current_topic}</li>
<li><strong>Application</strong>: Mise en pratique en {sujet}</li>
</ul>

**Exemples de code** :
{code_examples.get(sujet, f"// Exemple de code pour {current_topic}")}

Souhaitez-vous continuer ?""",
        
        "en": f"""**Spoken explanation** :
In this section on {current_topic}, we will explore the fundamental concepts of this topic in the context of {sujet}. This section is designed for a {niveau} level and will give you the necessary foundations to understand and apply these concepts. We will see the key elements and best practices to remember.

**HTML summary** :
<ul>
<li><strong>Objective</strong>: Understand {current_topic}</li>
<li><strong>Level</strong>: {niveau}</li>
<li><strong>Key points</strong>: Fundamental concepts of {current_topic}</li>
<li><strong>Application</strong>: Practice in {sujet}</li>
</ul>

**Code examples** :
{code_examples.get(sujet, f"// Code example for {current_topic}")}

Would you like to continue?""",
        
        "es": f"""**Explicación oral** :
En esta sección sobre {current_topic}, exploraremos los conceptos fundamentales de este tema en el contexto de {sujet}. Esta sección está diseñada para un nivel {niveau} y te dará las bases necesarias para comprender y aplicar estos conceptos. Veremos los elementos clave y las mejores prácticas a recordar.

**Resumen HTML** :
<ul>
<li><strong>Objetivo</strong>: Comprender {current_topic}</li>
<li><strong>Nivel</strong>: {niveau}</li>
<li><strong>Puntos clave</strong>: Conceptos fundamentales de {current_topic}</li>
<li><strong>Aplicación</strong>: Práctica en {sujet}</li>
</ul>

**Ejemplos de código** :
{code_examples.get(sujet, f"// Ejemplo de código para {current_topic}")}

¿Quieres continuar?""",
"it": f"""**Spiegazione orale** :
In questa sezione su {current_topic}, esploreremo i concetti fondamentali di questo argomento nel contesto di {sujet}. Questa sezione è pensata per un livello {niveau} e fornirà le basi necessarie per comprendere e applicare questi concetti. Vedremo gli elementi chiave e le migliori pratiche da ricordare.

**Riepilogo HTML** :
<ul>
<li><strong>Obiettivo</strong>: Comprendere {current_topic}</li>
<li><strong>Livello</strong>: {niveau}</li>
<li><strong>Punti chiave</strong>: Concetti fondamentali di {current_topic}</li>
<li><strong>Applicazione</strong>: Pratica in {sujet}</li>
</ul>

**Esempi di codice** :
{code_examples.get(sujet, f"// Esempio di codice per {current_topic}")}

Vuoi continuare?"""

    }
    
    return templates.get(lang, templates["fr"])


# Nouvelle fonction pour sauvegarder les slides en JSON
def save_slides_to_json(slides, filename):
    data = {"slides": slides}
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Slides sauvegardées dans {filename}")

def extract_explanation_only(text: str) -> str:
    """Extrait uniquement la partie Explication orale ou Spoken Explanation"""

    patterns = [
        r'\*\*Explication orale\s*\*\*\s*:?\s*(.+?)(?=\n\s*\*\*|$)',  # FR
        r'\*\*Spoken explanation\s*\*\*\s*:?\s*(.+?)(?=\n\s*\*\*|$)',  # EN
        r'\*\*Explicación oral\s*\*\*\s*:?\s*(.+?)(?=\n\s*\*\*|$)',    # ES
        r'\*\*Spiegazione orale\s*\*\*\s*:?\s*(.+?)(?=\n\s*\*\*|$)',  # IT

    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            explanation = match.group(1).strip()
            explanation = re.sub(r'\n+', ' ', explanation)
            return explanation

    return "Explication indisponible"


def extract_summary_only(text: str) -> str:
    """Extrait uniquement la partie Résumé HTML / HTML summary / Resumen HTML"""
    patterns = [
        r'\*\*Résumé HTML\*\*\s*:?\s*(<ul>.*?</ul>)',     # FR
        r'\*\*HTML summary\*\*\s*:?\s*(<ul>.*?</ul>)',    # EN
        r'\*\*Resumen HTML\*\*\s*:?\s*(<ul>.*?</ul>)',    # ES
        r'\*\*Riepilogo HTML\*\*\s*:?\s*(<ul>.*?</ul>)',  # IT

    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return "<ul><li>Résumé indisponible</li></ul>"


def extract_example_code_only(text: str) -> str:
    """Extrait uniquement la partie Exemples de code"""
    pattern = r'<pre><code.*?>.*?</code></pre>'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0).strip()
    return "<pre><code>// Pas d'exemple de code disponible</code></pre>"

# ==== Interface terminal améliorée ====
def main():
    print("\n📚 Formation interactive multilangue avec RAG & Ollama (LLaMA3)")
    print("Langues : [fr]ançais, [en]glish, [es]pagnol, [it]alien")
    print("Domaines disponibles :", ", ".join(AVAILABLE_DOMAINS.keys()))
    print("Tapez 'q' pour quitter.\n")

    # Vérifier Ollama
    if not check_ollama_status():
        print("❌ Ollama n'est pas accessible. Assurez-vous qu'il est démarré avec 'ollama serve'")
        return

    # Configuration
    lang = input("🌐 Langue [fr/en/es/it] : ").strip().lower()
    if lang not in SYSTEM_PROMPT:
        lang = "fr"

    sujet = input("📌 Sujet (ex: angular, java, jee) : ").strip().lower()
    
    if sujet not in AVAILABLE_DOMAINS:
        print(f"❌ Domaine '{sujet}' non disponible. Domaines disponibles : {', '.join(AVAILABLE_DOMAINS.keys())}")
        return

    niveau = input("🎓 Niveau (débutant / intermédiaire / avancé) : ").strip().lower()

    print("\n🧭 Donnez le plan de la formation.")
    print("Tapez les axes du plan sur une seule ligne séparés par des virgules,")
    print("ou entrez chaque axe sur une ligne différente, puis tapez une ligne vide pour terminer.\n")

    # Entrée du plan
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line.strip())

    if len(lines) == 1 and "," in lines[0]:
        plan_parts = [p.strip() for p in lines[0].split(",") if p.strip()]
        plan_input = ", ".join(plan_parts)
    else:
        plan_parts = lines
        plan_input = "\n".join(plan_parts)

    if not plan_parts:
        print("❌ Aucun plan fourni. Arrêt du programme.")
        return

    print(f"\n📋 Plan de formation validé avec {len(plan_parts)} parties:")
    for i, part in enumerate(plan_parts, 1):
        print(f"  {i}. {part}")

    # Génération des slides
    current_part_index = 0
    history = ""
    slide_number = 1
    spoken_data = []
    slides_data=[]

    print("\n🚀 Démarrage de la génération des slides...")
    print("💡 En cas de timeout, des slides de secours seront générées.\n")

    while current_part_index < len(plan_parts):
        current_part = plan_parts[current_part_index]
        print(f"\n📌 Partie {current_part_index + 1}/{len(plan_parts)} : {current_part}")
        print(f"🔢 Génération de la Slide {slide_number}")

        # Générer la slide
        question = f"Expliquer {current_part} pour {niveau} niveau en {sujet}"
        
        response_raw = rag_query(
            query=question,
            sujet=sujet,
            niveau=niveau,
            plan=plan_input,
            history=history,
            current_topic=current_part,
            slide_number=slide_number,
            lang=lang
        )
        
        # Vérifier si la génération a échoué
        if "❌" in response_raw and "Échec" in response_raw:
            print("🔄 Génération de slide de secours...")
            response_raw = generate_fallback_slide(current_part, slide_number, sujet, niveau, lang)
        
        response = f"🟩 Slide {slide_number}: {current_part}\n\n{response_raw.strip()}"
        print("\n📘 Réponse générée :\n")
        print(response)
        print("\n" + "="*80 + "\n")

        # Ajouter à l'historique
        history += f"\n\n--- SLIDE {slide_number}: {current_part} ---\n{response_raw.strip()}\n"
        explanation_only = extract_explanation_only(response_raw)
        spoken_data.append({
            "id": slide_number,
            "title": current_part,
            "script": explanation_only
        })

        summary_only = extract_summary_only(response_raw)
        example_code_only = extract_example_code_only(response_raw)

        slides_data.append({
            "id": slide_number,
            "title": current_part,
            "summary": summary_only,
            "example_code": example_code_only
        })

        slide_number += 1
        current_part_index += 1

        # Demander confirmation
        if current_part_index < len(plan_parts):
            try:
                continuer = input(f"\n🔁 Continuer vers la partie suivante ({plan_parts[current_part_index]}) ? [O/n] : ").strip().lower()
                if continuer == "n":
                    print("✅ Fin de la génération interrompue par l'utilisateur.")
                    break
            except KeyboardInterrupt:
                print("\n✅ Arrêt du programme par l'utilisateur.")
                break
        else:
            print("✅ Toutes les parties du plan ont été traitées.")
    save_slides_to_json(spoken_data,"Explanation Output/"+lang+"-explanation-"+sujet+".json")
    save_slides_to_json(slides_data, "Summary Output/"+lang+"-summary-code-"+sujet+".json")

    print("\n✅ Fin de la génération complète de la formation.")
    print(f"📊 Résumé : {slide_number - 1} slides générées sur {len(plan_parts)} parties planifiées.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n✅ Programme interrompu par l'utilisateur.")
    except Exception as e:
        print(f"\n❌ Erreur fatale: {str(e)}")