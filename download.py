# from bs4 import BeautifulSoup
# import requests

from selenium import webdriver
# from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import time
from lxml import html
import re
import os
import math

outputFolder = "out"
if not os.path.exists(outputFolder):
    os.mkdir(outputFolder)

def strip_html(s):
    return str(html.fromstring(s).text_content())

cpt = sum([len(files) for r, d, files in os.walk(outputFolder)])
toPage = max(0, math.floor(cpt / 10 - 5))
print("Files:", cpt)

toPage = 0

# options = Options()
# options.headless = True
# driver = webdriver.Firefox(options=options)
chromeOptions = webdriver.ChromeOptions()
chromeOptions.add_argument("--headless")
driver = webdriver.Chrome(options=chromeOptions)

while True:
    try:
        driver.get("https://scuolalavoro.registroimprese.it/rasl/resultSearch")

        thisPage = 1
        while thisPage != None:

            thisPageTmp = None
            while thisPageTmp != thisPage:
                # print("Waiting")
                time.sleep(1)
                paginations = driver.find_elements_by_css_selector(".pagination")
                ems = paginations[0].find_elements_by_css_selector("em span")
                for em in ems:
                    t = strip_html(em.get_attribute('outerHTML')).strip()
                    if len(t):
                        thisPageTmp = int(t)

            containers = driver.find_elements_by_css_selector("#content .container .bgWhite")
            for i in range(len(containers)):
                container = containers[i]

                cf = None
                # print(container.get_attribute('innerHTML'))
                cfs = container.find_elements_by_css_selector(".rowsmall .ten span")
                if len(cfs) > 0:
                    cf = cfs[0].get_attribute('innerHTML')

                if cf == None:
                    print("Unable to find CF")
                    continue
                print("CF:", cf)

                thisFolder = os.path.join(outputFolder, cf[:2])
                if not os.path.exists(thisFolder):
                    os.mkdir(thisFolder)

                thisFile = os.path.join(thisFolder, cf)
                if os.path.exists(thisFile):
                    print("This file already exists, skipping")
                    continue

                titles = container.find_elements_by_css_selector("#title h5 a")
                if len(titles) == 0:
                    print("Unable to find title for ", cf)
                    continue

                title = ""
                title = strip_html(titles[0].get_attribute('innerHTML'))
                title = title.strip()
                title = re.sub(r"\s+", " ", title)
                print("Title:", title)

                thisId = titles[0].get_attribute("id")
                # print("ID:", thisId)
                containerHtml = container.get_attribute('innerHTML')
                driver.execute_script("document.getElementById('" + thisId + "').click();")
                time.sleep(1)
                # driver.execute_script("window.scrollBy(0, -100);")
                # time.sleep(1)

                # titles[0].click()

                element = None
                try:
                    element = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.TAG_NAME, "h3"))
                    )
                    content = driver.find_element_by_css_selector("#content")

                    print("Saving information")
                    with open(thisFile, "w") as fw:
                        fw.write(content.get_attribute('innerHTML'))
                except Exception as e:
                    print("Unable to load page:", str(e))
                    with open(thisFile, "w") as fw:
                        fw.write("ERROR\n\n")
                        fw.write(containerHtml)
                    # print(driver.page_source)

                print("Going back")
                driver.execute_script("window.history.go(-1)")
                time.sleep(1)
                element = None
                try:
                    element = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "pagination"))
                    )
                except Exception as e:
                    print("Unable to go back:", str(e))
                    try:
                        driver.close()
                        exit()
                    except:
                        break

                containers = driver.find_elements_by_css_selector("#content .container .bgWhite")

            paginationsLinks = driver.find_elements_by_css_selector(".pagination a")
            linkId = ""
            pageNo = 0
            for link in paginationsLinks:
                spans = link.find_elements_by_css_selector("span")
                if len(spans):
                    pageNo = int(spans[0].get_attribute('innerHTML'))
                    linkId = link.get_attribute("id")
                    if pageNo == thisPage + 1 and toPage <= pageNo:
                        thisPage = pageNo
                        print("Going to page:", thisPage)
                        driver.execute_script("document.getElementById('" + linkId + "').click();")
                        time.sleep(1)
                        linkId = ""
                        break

            if len(linkId):
                thisPage = pageNo
                print("Going to page:", pageNo)
                driver.execute_script("document.getElementById('" + linkId + "').click();")
                time.sleep(1)
    finally:
        try:
            driver.close()
        finally:
            pass





exit()

time.sleep(5)

content = driver.find_elements_by_css_selector("#content")
print(content[0].get_attribute('outerHTML'))

exit()

for title in titles:
    print(title.get_attribute('outerHTML'))

paginations = driver.find_elements_by_css_selector(".pagination")
nextPage = paginations[1].find_element_by_xpath("//a[@title='Go to page 2']")
print(nextPage.get_attribute('outerHTML'))

nextPage.click()

print("Waiting")
time.sleep(10)

# print(nextPage.get_attribute('outerHTML'))

titles = driver.find_elements_by_css_selector("#content .container #title h5 .linkHeader")
for title in titles:
    print(title.get_attribute('outerHTML'))

driver.close()

exit()

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

driver = webdriver.Firefox()
driver.get("https://scuolalavoro.registroimprese.it/rasl/resultSearch")
# assert "Python" in driver.title
# elem = driver.find_element_by_name("q")
# elem.clear()
# elem.send_keys("pycon")
# elem.send_keys(Keys.RETURN)
# assert "No results found." not in driver.page_source
driver.close()


exit()

main_url = "https://scuolalavoro.registroimprese.it/rasl/resultSearch"
req = requests.get(main_url)
soup = BeautifulSoup(req.text, "html.parser")

# print(soup.prettify())

# titles = soup.select("#content .container #title h5 .linkHeader")
# for title in titles:
#     print(title.get_text().strip())

paginations = soup.select(".pagination")
for pagination in paginations:
    print(pagination)

exit()

contents = soup.find_all(id="content")
for content in contents:
    divs = content.find_all("div", class_="container")
    for div in divs:
        title = div.find("#title h5")
        if title:
            print(title.get_text().strip())
