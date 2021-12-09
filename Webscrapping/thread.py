from selenium import webdriver
import pandas as pd
import concurrent.futures
import time
import re
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import Future 

urls = []
titles = []

product = pd.read_csv(r"C:\\Users\\LENOVO\\Downloads\\shein-baglallery-prices.csv")
product = product.head(n=2)


product = product.reset_index(drop=True)
skus = product['SKU_new']

#to make url by skus
urls= []
print("start")
for sku in skus:
    urls.append("https://eur.shein.com/pdsearch/"+sku+"/?ici=s1%60RecentSearch%60"+sku+"%60_fb%60d0%60PageSearchResult&scici=Search~~RecentSearch~~1~~"+sku+"~~~~0")
  

global driver
global output
output = []

opts = webdriver.ChromeOptions()
driver = webdriver.Chrome(options= opts, executable_path = 'chromedriver.exe')

start = time.time()


def check(prices):


    return

def transform(url_shein):
    print("url =>",url_shein)
    driver.get(url_shein)
    prices = driver.find_elements_by_class_name('S-product-item__retail-price')
    
    if not prices:
        output.append('not found')

    for price in prices:
        output.append(price.text)
    
    print(price.text)
    return prices


#multithreading 
# with concurrent.futures.ThreadPoolExecutor() as executor:
#     executor.map(transform, urls)

test = []
# with concurrent.futures.ThreadPoolExecutor() as executor:
#     future_kaid = {executor.submit(transform,url): url for url in urls}
#     for future in concurrent.futures.as_completed(future_kaid):
#         kaid = future_kaid[future]
#         data = future.result()
#         test.append(data[0])


with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [ executor.submit(transform, url) for url in urls]
    results = []
    for result in concurrent.futures.as_completed(futures):
        results.append(result)

# for url in urls:
#     transform(url)



print("test => ", result)
driver.quit()
