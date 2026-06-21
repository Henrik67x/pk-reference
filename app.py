import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from google import genai
import time

import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
NST_KEY = os.getenv("NST_KEY")

client = genai.Client(api_key=GEMINI_KEY)

def ask_gemini(prompt):
    response = client.models.generate_content(
        model="models/gemini-3-flash-preview",
        contents=prompt
    )
    return response.text

def get_nhl_data(player_name):
    search_url = f"https://search.d3.nhle.com/api/v1/search/player?culture=en-us&limit=5&q={player_name.replace(' ', '%20')}&active=true"
    response = requests.get(search_url)
    data = response.json()
    if not data:
        return None
    player_id = data[0]["playerId"]
    player_url = f"https://api-web.nhle.com/v1/player/{player_id}/landing"
    return requests.get(player_url).json()

def get_nst_data(player_name, season="20252026"):
    search_url = f"https://search.d3.nhle.com/api/v1/search/player?culture=en-us&limit=5&q={player_name.replace(' ', '%20')}&active=true"
    response = requests.get(search_url)
    data = response.json()
    if not data:
        return None
    player_id = data[0]["playerId"]
    url = f"https://data.naturalstattrick.com/playerreport.php?fromseason={season}&thruseason={season}&stype=2&sit=ev&stdoi=std&rate=n&v=p&playerid={player_id}&key={NST_KEY}"
    nst_response = requests.get(url)
    soup = BeautifulSoup(nst_response.content, "html.parser")
    return soup.get_text()[:5000]

def get_elite_prospects_data(player_name):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    query = f"eliteprospects.com/player {player_name}"
    driver.get(f"https://duckduckgo.com/?q={query.replace(' ', '+')}")
    time.sleep(4)
    ep_url = None
    for link in driver.find_elements(By.TAG_NAME, "a"):
        href = link.get_attribute("href")
        if href and "eliteprospects.com/player/" in href:
            ep_url = href
            break
    if not ep_url:
        driver.quit()
        return None
    driver.get(ep_url)
    time.sleep(4)
    text = driver.find_element(By.TAG_NAME, "body").text[:10000]
    driver.quit()
    return text

def extract_player_name(question):
    return ask_gemini(f"Extract the NHL player name from this question. Return ONLY the player's full name, nothing else. If no specific player is mentioned, return 'none'. Question: {question}").strip()

def answer_question(user_question):
    player_name = extract_player_name(user_question)
    if player_name.lower() == "none":
        return "Please ask about a specific NHL player."
    print(f"Fetching data for: {player_name}")
    nhl_data = get_nhl_data(player_name)
    nst_data = get_nst_data(player_name)
    ep_data = get_elite_prospects_data(player_name)
    return ask_gemini(f"""A user asked: "{user_question}"

Here is live NHL API data for {player_name}:
{nhl_data}

Here is Natural Stat Trick advanced stats for {player_name}:
{nst_data}

Here is Elite Prospects data for {player_name}:
{ep_data}

Answer the question using all three data sources. Follow these rules strictly:
- Be professional, concise, and data-driven. No fluff or filler sentences.
- Never state obvious hockey context as if it is insight
- Never make up data. If something is not in the data provided, say it is not available.
- Lead with the specific data the user asked for
- Use short summary sentences only where genuinely useful
- Format stats in clean tables where appropriate
- Tone should match a professional hockey analyst, not a fan blogger
- Do not cut any information off""")