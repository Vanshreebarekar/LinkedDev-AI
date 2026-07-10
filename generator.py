import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment configurations from .env file
load_dotenv()


class LinkedInPostGenerator:
    def __init__(self):
        """
        Initializes the Gemini client instance safely using system environment variables
        with a direct fallback string using the modern Google GenAI SDK.
        """
        # 1. Try to read from environment variables first
        api_key = os.getenv("GEMINI_API_KEY")

        # 2. FALLBACK: If the environment variable is missing or broken, paste your key below
            api_key = "YOUR_API_KEY_HERE"

        if not api_key:
            raise ValueError("Missing Configuration: GEMINI_API_KEY could not be verified.")

        # FIXED: Initializing the modern client and model name variables cleanly
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.5-flash"

    def _extract_and_chunk_article(self, url: str) -> str:
        """
        Downloads the webpage or raw code content directly and chunks text safely.
        Includes a dynamic URL rewriter to safely handle GitHub rate blocks.
        """
        try:
            target_url = url

            # --- GITHUB BYPASS LOGIC ---
            if "github.com" in target_url and "/blob/" in target_url:
                print("[System] GitHub link detected. Converting to Raw content URL to bypass 429 blocks...")
                target_url = target_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

            print(f"[System] Extracting content from: {target_url}...")

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }

            response = requests.get(target_url, headers=headers, timeout=15)
            response.raise_for_status()

            if target_url.endswith(".py") or "raw.githubusercontent" in target_url:
                full_text = response.text
            else:
                soup = BeautifulSoup(response.text, "html.parser")
                paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
                full_text = "\n\n".join(paragraphs)
                if len(full_text) < 200:
                    full_text = soup.body.get_text(separator=" ", strip=True)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(full_text)

            context_text = "\n\n".join(chunks)
            return context_text[:6000]

        except Exception as e:
            print(f"[Warning] Web extraction encountered an issue: {e}")
            return ""

    def generate_post(self, url: str) -> str:
        """
        Processes context and coordinates with Gemini to output a highly engaging post.
        """
        context = self._extract_and_chunk_article(url)
        if not context:
            return "Failed to retrieve or process data from the provided URL."

        repo_name = url.split("/")[-1] if "/" in url else "my-project"
        if "?" in repo_name:
            repo_name = repo_name.split("?")[0]

        # Tailored Developer & Data Science Prompt Framework
        prompt = f"""
        You are a passionate Software Engineer and Data Scientist sharing your authentic building journey on LinkedIn. 
        Your task is to review the following project context and write an engaging, engineering-focused LinkedIn update. Do NOT invent fake marketing or business use cases unless explicitly mentioned in the text.

        Project Context:
        \"\"\"
        {context}
        \"\"\"

        Formatting Rules:
        1. Hook: Start with an authentic engineering problem statement, a fascinating data insight, or a proud project milestone.
        2. Technical Core: Highlight 3 precise technical implementations from the project and explain their specific engineering purpose. Use code emojis like 💻, 📊, 🔍, ⚙️ for bullets.
        3. Tone: Passionate, developer-centric, clean, and highly conversational. Completely avoid boring corporate marketing fluff.
        4. Call to Action: End with an engaging technical question that invites other software engineers to comment or share insights.
        5. Personal Branding: The project reference repository link at the end MUST always point to my personal profile user handle 'Vanshreebarekar'. Automatically rewrite the final URL layout to display exactly like this: https://github.com/Vanshreebarekar/{repo_name}
        6. Length & Extras: Keep it crisp and concise (under 180 words total). Wrap up with 3-4 highly precise hashtags (e.g., #MachineLearning #ComputerVision #Python).
        """

        print(f"[System] Prompting {self.model_name} for creative copy drafting...")

        try:
            # Fully synchronized call structure
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                )
            )
            return response.text
        except Exception as e:
            return f"API Error during generation: {e}"
