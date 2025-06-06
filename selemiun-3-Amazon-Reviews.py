import time
import json
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException, StaleElementReferenceException

def create_driver():
    options = Options()
    options.add_argument("--user-data-dir=C:/Users/Vasundhra Bhatt/AppData/Local/Google/Chrome/User Data")
    options.add_argument("--profile-directory=Default")  # or "Profile 1", etc.
    options.add_argument("--start-maximized")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    service = Service("C:\\Program Files (x86)\\chromedriver.exe")
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def load_json(file_path):
    with open(file_path, "r") as json_file:
        return json.load(json_file)

def save_json(data, file_path):
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

def update_review_in_json(product_url, review_id, review_data, file_path="amazon_search_results_1.json"):
    with open(file_path, "r+") as json_file:
        data = json.load(json_file)
        for product in data:
            if product["link"] == product_url:
                if "Reviewer Details" not in product:
                    product["Reviewer Details"] = {}
                product["Reviewer Details"][review_id] = review_data
                break
        json_file.seek(0)
        json.dump(data, json_file, indent=4)
        json_file.truncate()

def collect_reviews(driver, product_url):
    product_data = {}
    reviews_data = {}
    driver.get(product_url)
    time.sleep(10)  # Wait for the page to load

    # Collect product title
    product_title = driver.find_element(By.ID, "productTitle").text.strip()
    print("Collecting reviews for:", product_title)

    # Collect the text in the product description
    try:
        product_description_element = driver.find_element(By.XPATH, "//*[@id='productDescription']/p/span")
        product_description = product_description_element.text
        product_data['product_description'] = product_description
    except Exception as e:
        print(f"An error occurred while fetching product_description: {e}")

    # Collect the text in the product summary
    try:
        product_summary_element = driver.find_element(By.XPATH, "//*[@id='feature-bullets']")
        product_summary = product_summary_element.text
        product_data['product_summary'] = product_summary
    except Exception as e:
        print(f"An error occurred while fetching product_summary: {e}")

    # Click on the customer reviews section
    customer_reviews_section = driver.find_element(By.ID, "acrCustomerReviewText")
    ActionChains(driver).move_to_element(customer_reviews_section).click().perform()

    # Collect review summary data
    reviews_medley = driver.find_element(By.ID, "reviewsMedley")
    review_elements = reviews_medley.find_elements(By.CLASS_NAME, "a-fixed-left-grid-inner")

    for review in review_elements:
        col_left = review.find_element(By.CSS_SELECTOR, ".a-fixed-left-grid-col.a-col-left")
        child_elements_left = col_left.find_elements(By.XPATH, ".//*")

        for child in child_elements_left:
            if "cr-widget-TitleRatingsHistogram" in child.get_attribute("class"):
                star_rating_element = child.find_element(By.CSS_SELECTOR, "._cr-ratings-histogram_style_star-rating__s2nPF")
                text_content = star_rating_element.text.split('\n')
                
                reviews_data['Average reviews'] = text_content[1]
                reviews_data['Total Ratings'] = text_content[2]
                reviews_per_star = {
                    "5 star": text_content[4],
                    "4 star": text_content[6],
                    "3 star": text_content[8],
                    "2 star": text_content[10],
                    "1 star": text_content[12]
                }
                reviews_data['reviews_per_star'] = reviews_per_star

    # Update product info and review summary
    update_product_info(product_url, product_data, reviews_data)

    # Collect individual reviews
    see_more_reviews_link = driver.find_element(By.XPATH, "//a[@data-hook='see-all-reviews-link-foot']")
    ActionChains(driver).move_to_element(see_more_reviews_link).click().perform()
    time.sleep(4)  # Wait for the page to load

    while True:
        review_list = driver.find_element(By.ID, "cm_cr-review_list")
        child_divs = review_list.find_elements(By.XPATH, ".//div")
        try: 
            for div in child_divs:
                print("got child divs and in the for loop")
                div_id = div.get_attribute('id')
                if div_id and '-review-card' in div_id:
                    print("in if loop")
                    rest_of_string = div_id.replace('-review-card', '')
                    new_id = f"customer_review-{rest_of_string}"
                    review_details = {}
                    time.sleep(1)
                    
                    try:
                        print("in 1st try block")
                        review_element_1 = driver.find_element(By.XPATH, f"//*[@id='{new_id}']/div[1]/a/div[2]/span")
                        review_details['reviewer_name'] = review_element_1.text
                        
                        review_element_2 = driver.find_element(By.XPATH, f"//*[@id='{new_id}']/div[2]/h5/a/i/span")
                        review_details['review_stars'] = review_element_2.get_attribute('textContent')
                        
                        review_element_3 = driver.find_element(By.XPATH, f"//*[@id='{new_id}']/div[2]/h5/a/span[2]")
                        review_details['review_title'] = review_element_3.text
                        
                        review_element_4 = driver.find_element(By.XPATH, f"//*[@id='{new_id}']/span")
                        review_details['review_date'] = review_element_4.text
                        
                        review_element_5 = driver.find_element(By.XPATH, f"//*[@id='{new_id}']/div[4]/span/span")
                        review_details['review'] = review_element_5.text
                        
                        # Update MongoDB with this review immediately
                        update_review_in_json(product_url, new_id, review_details)
                        print(f"Updated review: {new_id}")
                    except Exception as e:
                        print(f"Error processing review details: {e}")
                        try:
                            new_id = f"customer_review_foreign-{rest_of_string}"
                            review_element_1 = driver.find_element(By.XPATH, f"//*[@id='{new_id}']/div[1]/div/div[2]/span")
                            review_details['reviewer_name'] = review_element_1.text
                            
                            review_element_2 = driver.find_element(By.XPATH, f"//*[@id='{new_id}']/div[2]/h5/i/span")
                            review_details['review_stars'] = review_element_2.get_attribute('textContent')
                            
                            review_element_3 = driver.find_element(By.XPATH, f"//*[@id='{new_id}']/div[2]/h5/span[2]/span")
                            review_details['review_title'] = review_element_3.text
                            
                            review_element_4 = driver.find_element(By.XPATH, f"//*[@id='{new_id}']/span")
                            review_details['review_date'] = review_element_4.text
                            
                            review_element_5 = driver.find_element(By.XPATH, f"//*[@id='{new_id}']/div[4]/span/span")
                            review_details['review'] = review_element_5.text
                            
                            # Update MongoDB with this review immediately
                            update_review_in_json(product_url, new_id, review_details)
                            print(f"Updated review: {new_id}")
                        except Exception as e:
                            print(f"Error processing review: {e}")
                            continue
        except Exception as e:
            print(f"An error occurred while fetching reviews: {e}")
            continue
        try:
            next_page_link = driver.find_element(By.XPATH, "//*[@id='cm_cr-pagination_bar']/ul/li[2]/a")
            next_page_link.click()
            time.sleep(2)  # Wait for the next page to load
        except:
            print("No more pages available.")
            break

def update_product_info(product_url, product_data, reviews_data, file_path="amazon_search_results_1.json"):
    with open(file_path, "r+") as json_file:
        data = json.load(json_file)
        for product in data:
            if product["link"] == product_url:
                product["Product Description"] = product_data
                product["Review Summary"] = reviews_data
                break
        json_file.seek(0)
        json.dump(data, json_file, indent=4)
        json_file.truncate()

def main():
    driver = create_driver()

    driver.get("https://www.amazon.com/")
    print("🔐 Please click 'Sign In' and log into your Amazon account manually.")
    input("After logging in completely, press Enter here to continue...") 


    # Load the existing JSON file
    products = load_json("amazon_search_results_1.json")



    for product in products:
        product_url = product["link"]
        print(f"Collecting reviews for: {product_url}")
        try:
            collect_reviews(driver, product_url)
            print(f"Finished processing {product_url}")
        except (NoSuchElementException, TimeoutException, WebDriverException, StaleElementReferenceException) as e:
            print(f"Error processing {product_url}: {e}")
            continue  # Move to the next product URL

    driver.quit()

    print("All products have been processed and the JSON file has been updated.")

if __name__ == "__main__":
    main()