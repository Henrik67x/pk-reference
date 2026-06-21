from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time

def get_prospect_data(player_name):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    
    # Step 1 - find the EP URL via DuckDuckGo
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
        return "Could not find player on Elite Prospects"
    
    print(f"Found: {ep_url}")
    
    # Step 2 - scrape the player page
    driver.get(ep_url)
    time.sleep(4)
    
    text = driver.find_element(By.TAG_NAME, "body").text[:3000]
    driver.quit()
    return text

if __name__ == "__main__":
    result = get_prospect_data("Leo Carlsson")
    print(result)