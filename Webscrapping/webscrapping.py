from selenium import webdriver
import pandas as pd

product = pd.read_csv(r"C:\\Users\\LENOVO\Downloads\\R_Check2\\products_export.csv")
for i in range(len(product)):
  index = product['Variant SKU'].loc[i].find('-')
  product['Variant SKU'].loc[i] = product['Variant SKU'].loc[i][:index]

product = product.drop_duplicates(subset = ['Variant SKU'])
product = product.reset_index(drop=True)

opts = webdriver.ChromeOptions()
# The below line will make your browser run in background when uncommented
# opts.add_argument('--headless')

driver = webdriver.Chrome(options= opts, executable_path = 'chromedriver.exe')

products = product['Variant SKU']
output = []

for i in range(len(products)):

    url_shein = url = "https://eur.shein.com/pdsearch/"+products[i]+"/?ici=s1%60RecentSearch%60"+products[i]+"%60_fb%60d0%60PageSearchResult&scici=Search~~RecentSearch~~1~~"+products[i]+"~~~~0"

    driver.get(url_shein)
    prices = driver.find_elements_by_class_name('S-product-item__retail-price')


    if not prices:
        output.append('not found')


    for price in prices:
        output.append(price.text)


driver.quit()


d = {'Variant SKU':products,'price':output}
df = pd.DataFrame(d)
print(df)
