from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def test_browser_prediction():
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    
    try:
        driver.get("http://localhost:8501")
        # Wait up to 10 seconds for the app to actually load
        wait = WebDriverWait(driver, 10)

        # STEP 1: Click the "Diagnostics" Tab first! 
        # (The robot starts on 'Overview', but the button is in 'Diagnostics')
        diag_tab = wait.until(EC.element_to_be_clickable((By.XPATH, "//p[contains(text(), 'Diagnostics')]")))
        diag_tab.click()
        time.sleep(2) # Give the tab a second to switch

        # STEP 2: Find and click the Predict button
        # Streamlit buttons are usually inside a <button> tag
        predict_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Predict Death Event Risk')]")))
        predict_button.click()
        print("✅ Robot successfully clicked the Predict button!")

        # STEP 3: Check if the Gauge appeared
        time.sleep(3)
        if "Probability" in driver.page_source:
            print("🎉 TEST PASSED: Prediction results are visible!")
        else:
            print("❌ TEST FAILED: Could not find results.")

    except Exception as e:
        print(f"⚠️ Error occurred: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    test_browser_prediction()