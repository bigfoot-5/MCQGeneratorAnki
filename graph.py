import re
import os
import json
from typing import TypedDict, List, Dict, Optional, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

try:
    from config import (
        API_KEY, AI_MODEL, OLLAMA_MODEL, OLLAMA_URL, 
        TEMPERATURE, LLM_PROVIDER,
        PROMPT_TEMPLATE, SYNONYM_PROMPT, EXPLANATION_PROMPT,
        NEWS_BASE_PARAMS, NEWS_API_URL
    )
    # Re-use news fetcher logic if available
    from news_fetcher import fetch_news as newsapi_fetch_news
except ImportError:
    # Fallback or Mocking if strictly necessary, but config should exist now
    pass

# --- 1. State Definition ---
class QuizState(TypedDict):
    word: str
    definition: str
    article: Optional[Dict]
    sentence: Optional[str]
    sentence_masked: Optional[str]
    synonyms: List[str]
    explanation: Optional[str]
    word_length: int
    first_letter: str
    # Flags
    sentence_generated: bool
    error: Optional[str]


# --- 2. Node Implementations ---

def get_llm():
    """Factory to get the configured LLM"""
    if LLM_PROVIDER == "ollama":
        if not OLLAMA_MODEL:
            raise ValueError("OLLAMA_MODEL not configured")
        # ChatOllama usually takes base_url
        base_url = OLLAMA_URL.replace("/api/chat", "") # rough fix if needed, ChatOllama default is http://localhost:11434
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=base_url,
            temperature=TEMPERATURE
        )
    else:
        if not API_KEY:
            raise ValueError("OPENAI_API_KEY not configured")
        return ChatOpenAI(
            api_key=API_KEY,
            model=AI_MODEL,
            temperature=TEMPERATURE
        )

# -- Node 2: Article Retriever --
def retrieve_article_node(state: QuizState) -> QuizState:
    """
    Retrieves news article. 
    In the previous implementation, articles were passed in from the runner.
    Here, we can simulate retrieval or use the one passed in initial state.
    For this 'test_main.py' context, we usually load all articles upfront.
    We will assume 'article' might already be in state, or we fetch one if missing.
    """
    print(f"--- [Node 2] Article Retriever for '{state['word']}' ---")
    
    # If article is already provided (e.g. by the caller), just return.
    if state.get("article"):
        print(f"    Using existing article: {state['article'].get('title', 'Unknown')}")
        return state
        
    # Otherwise, we could fetch here. For simplicity in this refactor, 
    # we'll assume the caller (test_main) injects it, or we leave it empty.
    # If we truly want to move logic here:
    print("    No article provided in state. Proceeding without article.")
    return {"article": None}


# -- Node 3: News Word Merger (Sentence Gen) --
def generate_sentence_with_article_node(state: QuizState) -> QuizState:
    print(f"--- [Node 3] News Word Merger (Sentence Gen) ---")
    
    word = state["word"]
    article = state.get("article")
    
    # If no article, we can't do "News Word Merger", so we fail this node 
    # and let conditional edge pick it up to go to Fallback.
    if not article:
        print("    No article found. Skipping to fallback.")
        return {"sentence_generated": False}
        
    definition = state.get("definition")
    
    # Build prompt
    article_title = article.get("title", "No title")
    article_summary = article.get("description") or article.get("summary") or "No summary"
    
    # We reconstruct the prompt using the template
    # NOTE: We can use LangChain prompts, but re-using the f-string template from config 
    # preserves exact behavior.
    clean_definition = (definition or "").replace("\n", " ")
    
    # Prepare prompt text
    prompt_text = PROMPT_TEMPLATE.format(
        word=word,
        inflection="plural", # Default
        definition=clean_definition,
        article_title=article_title,
        article_summary=article_summary
    )
    
    llm = get_llm()
    try:
        # Simple retry loop (2 attempts)
        for i in range(2):
            response = llm.invoke([HumanMessage(content=prompt_text)])
            content = response.content.strip()
            
            if word.lower() in content.lower():
                print("    Sentence generated successfully with article.")
                return {"sentence": content, "sentence_generated": True}
            else:
                print(f"    (Attempt {i+1}) Target word missing from sentence.")
    except Exception as e:
        print(f"    Error in generation: {e}")
        
    print("    Failed to generate valid sentence with article.")
    return {"sentence_generated": False}


# -- Node 4: Own Word Generator (Fallback) --
def generate_sentence_fallback_node(state: QuizState) -> QuizState:
    print(f"--- [Node 4] Own Word Generator (Fallback) ---")
    
    word = state["word"]
    definition = state.get("definition", "")
    
    # Simpler prompt for generating a sentence without article constraint
    fallback_prompt = f"""
    Write a single sentence using the word "{word}".
    Definition: {definition}
    The sentence should be clear, educational, and suitable for a vocabulary quiz.
    Context: "General Knowledge"
    """
    
    llm = get_llm()
    try:
        response = llm.invoke([HumanMessage(content=fallback_prompt)])
        content = response.content.strip()
        
        # Validation
        if word.lower() in content.lower():
             print("    Fallback sentence generated.")
             return {"sentence": content, "sentence_generated": True}
        else:
             # Last ditch effort: Just return the definition or a placeholder?
             # Or just the content and hope for the best
             print("    Fallback sentence missing word. Using content anyway.")
             return {"sentence": content, "sentence_generated": True}
             
    except Exception as e:
        print(f"    Fallback generation failed: {e}")
        return {"error": str(e), "sentence_generated": False}


# -- Helper: Masking Logic --
def _mask_sentence(word: str, sentence: str) -> str:
    word_len = len(word)
    if word_len == 0: return sentence
    
    escaped = re.escape(word)
    pattern = r'\b' + escaped + r'\b'
    match = re.search(pattern, sentence, flags=re.IGNORECASE)
    
    mask_text = f"{word[0]} ({len(word)})" # Default mask
    
    if match:
        actual_word = match.group(0)
        # Preserve first letter case from actual usage
        mask_text = f"{actual_word[0]} ({len(actual_word)})"
        return re.sub(pattern, mask_text, sentence, count=1, flags=re.IGNORECASE)
    elif word.lower() in sentence.lower():
        # Loose match
        pattern_loose = re.compile(re.escape(word), re.IGNORECASE)
        return pattern_loose.sub(mask_text, sentence, count=1)
    return sentence
    

# -- Node 1/5: Synonym Generator (and Explanation) --
# NOTE: User asked for "Node 1: Synonym Generator". 
# But Synonyms need the context of the sentence (Node 3) usually?
# "Node 1: Word Synonym generator - IT generates different synonyms for the given word."
# If it's context-independent properties, it can be first.
# But existing logic used 'sentence' context. I will put it AFTER sentence generation
# to maintain quality, as per the flow diagram in plan.
def generate_synonyms_and_extras_node(state: QuizState) -> QuizState:
    print(f"--- [Node 1/5] Synonym & Explanation Generator ---")
    
    if not state.get("sentence"):
        print("    No sentence available. Skipping.")
        return state
        
    word = state["word"]
    sentence = state["sentence"]
    definition = state.get("definition")
    
    llm = get_llm()
    
    # 1. Synonyms
    syn_prompt = SYNONYM_PROMPT.format(
        word=word, 
        sentence=sentence, 
        definition=definition or "", 
        num_synonyms=4
    )
    
    synonyms = []
    try:
        res = llm.invoke([HumanMessage(content=syn_prompt)])
        # Reuse parsing logic from generators.py (simplified here)
        content = res.content.strip()
        # Quick JSON parse attempt
        try:
             import json
             start = content.find("[")
             end = content.rfind("]")
             if start != -1 and end != -1:
                 synonyms = json.loads(content[start:end+1])
        except:
             synonyms = [p.strip("- ").strip() for p in content.splitlines()][:4]
    except Exception as e:
        print(f"    Synonym gen failed: {e}")
        
    # 2. Explanation
    exp_prompt = EXPLANATION_PROMPT.format(
        sentence=sentence,
        correct_word=word,
        definition=definition or ""
    )
    explanation = ""
    try:
        res = llm.invoke([HumanMessage(content=exp_prompt)])
        explanation = res.content.strip()
    except:
        pass

    # 3. Masking
    sentence_masked = _mask_sentence(word, sentence)
    
    return {
        "synonyms": synonyms,
        "explanation": explanation,
        "sentence_masked": sentence_masked,
        "word_length": len(word),
        "first_letter": word[0] if word else ""
    }


# -- Node 5: Output Maker --
def make_output_node(state: QuizState) -> QuizState:
    print(f"--- [Node 5] Output Maker ---")
    # This node essentially just passes state through, 
    # but could formatting logging or final validation.
    # In LangGraph, the final state IS the output, so this is just a logical checkpoint.
    return state


# --- 3. Graph Construction ---

def create_quiz_graph():
    workflow = StateGraph(QuizState)
    
    # Add Nodes
    workflow.add_node("retrieve_article", retrieve_article_node)
    workflow.add_node("news_word_merger", generate_sentence_with_article_node)
    workflow.add_node("own_word_generator", generate_sentence_fallback_node)
    workflow.add_node("synonym_generator", generate_synonyms_and_extras_node)
    workflow.add_node("output_maker", make_output_node)
    
    # Add Edges
    workflow.set_entry_point("retrieve_article")
    workflow.add_edge("retrieve_article", "news_word_merger")
    
    # Conditional Edge from Merger
    def merger_route(state: QuizState):
        if state.get("sentence_generated"):
            return "synonym_generator"
        else:
            return "own_word_generator"

    workflow.add_conditional_edges(
        "news_word_merger",
        merger_route,
        {
            "synonym_generator": "synonym_generator",
            "own_word_generator": "own_word_generator"
        }
    )
    
    workflow.add_edge("own_word_generator", "synonym_generator")
    workflow.add_edge("synonym_generator", "output_maker")
    workflow.add_edge("output_maker", END)
    
    return workflow.compile()
