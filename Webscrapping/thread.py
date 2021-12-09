from selenium import webdriver
import pandas as pd
import concurrent.futures
import time
import re
import numpy as np


urls = []
titles = []

product = pd.read_csv(r"C:\\Users\\LENOVO\\Downloads\\shein-baglallery-prices.csv")

# for i in range(len(product)):
#   index = product['Variant SKU'].loc[i].find('-')
#   product['Variant SKU'].loc[i] = product['Variant SKU'].loc[i][:index]

# product = product.drop_duplicates(subset = ['Variant SKU'])
# product['Variant SKU'] = product['Variant SKU'].str.split('-').str[0]
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

def transform(url_shein):
    print("url =>",url_shein)
    driver.get(url_shein)
    prices = driver.find_elements_by_class_name('S-product-item__retail-price')

    if not prices:
        output.append('not found')

    for price in prices:
        print(price.text)
        output.append(price.text)

    return



# for url in urls:
#     transform(url)

with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(transform, urls)

driver.quit()
end = time.time()
print("time taken for web scrapping => ",end - start)

print(output)
