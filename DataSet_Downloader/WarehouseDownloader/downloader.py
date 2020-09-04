from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from time import sleep
import zipfile
from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import requests
import shutil

DOWNLOADS_DIR = ""
WORKING_DIR = ""

EMAIL = ""
PASSWORD = ""

class Warehouse():
    def __init__(self):
        self.driver = webdriver.Chrome(chrome_options = self.getOptions())

        self.open()
        self.close_popups()
        self.log_in()

    def create_dirs(self):
        os.makedirs(os.path.join(DOWNLOADS_DIR,"tmp"), exist_ok=True)
        os.makedirs(os.path.join(WORKING_DIR,self.category), exist_ok=True)
        model_path = os.path.join(WORKING_DIR,self.category, "models.txt")
        if not os.path.exists(model_path):
            with open(model_path, 'a'):
                os.utime(model_path, None)

    def delete_tmp_folder(self):
        tmp_path = os.path.join(DOWNLOADS_DIR,"tmp")
        if os.path.isdir(tmp_path):
            shutil.rmtree(tmp_path)


    def getOptions(self):
        chromeOptions = webdriver.ChromeOptions()
        # chrome_options.add_argument("--kiosk") # FOR FULLSCREEN
        prefs = {"download.default_directory" : os.path.join(DOWNLOADS_DIR, "tmp")}
        chromeOptions.add_experimental_option("prefs",prefs)
        return chromeOptions

    def open(self):
        self.driver.get("https://3dwarehouse.sketchup.com/search/")
        sleep(3)

    def load_category(self):
        self.driver.get("https://3dwarehouse.sketchup.com/search/?q={}".format(self.category))
        sleep(3)

    def close_popups(self):
        try:
            welcome_btn = self.driver.find_element_by_xpath('//*[@id="sketchup2020promo-close"]')
            welcome_btn.click()
        except Exception:
            pass

        try:
            privacy_btn = self.driver.find_element_by_xpath('/html/body/div[7]/div[1]/div/a')
            privacy_btn.click()
        except Exception:
            pass


    def log_in(self):
        profile_image = self.driver.find_element_by_xpath('//*[@id="sign-in-menu-container"]/div/a')
        hover = ActionChains(self.driver).move_to_element(profile_image)
        hover.perform()

        login_btn = self.driver.find_element_by_xpath('//*[@id="scroller"]/div/button')
        login_btn.click()
        sleep(5)
        email_in = self.driver.find_element_by_xpath('//*[@id="email"]')
        email_in.send_keys(EMAIL)

        next_btn = self.driver.find_element_by_xpath('//*[@id="next"]')
        next_btn.click()
        sleep(0.5)

        pw_in = self.driver.find_element_by_xpath('//*[@id="password"]')
        pw_in.send_keys(PASSWORD)

        send_btn = self.driver.find_element_by_xpath('//*[@id="submit"]')
        send_btn.click()

        sleep(5)

    def scroll_to_bottom(self):
        SCROLL_PAUSE_TIME = 3

        # Get scroll height
        last_height = self.driver.execute_script("return document.body.scrollHeight")

        while True:
            try:
                # bottom load more button
                btn_loadmore = self.driver.find_element_by_xpath('//*[@id="scrollLimit"]')
                btn_loadmore.click()
                sleep(5)
            except Exception:
                pass

            # Scroll down to bottom
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            # Wait to load page
            sleep(SCROLL_PAUSE_TIME)

            # Calculate new scroll height and compare with last scroll height
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    def few_questions_dialog(self):
        industry = self.driver.find_element_by_xpath('//*[@id="industryDropdown"]/input')
        industry.click()

        student_dropdown = self.driver.find_element_by_xpath('//*[@id="industryDropdown"]/ul/li[8]/a')
        student_dropdown.click()

        language = self.driver.find_element_by_xpath('//*[@id="languageDropdown"]/input')
        language.click()

        german_dropdown = self.driver.find_element_by_xpath('//*[@id="languageDropdown"]/ul/li[2]/a')
        german_dropdown.click()

        next_btn = self.driver.find_element_by_xpath('//*[@id="required-fields-modal"]/div[2]/button')
        next_btn.click()

        sleep(0.5)

        interest = self.driver.find_element_by_xpath('//*[@id="required-fields-modal"]/div[1]/div[2]/div[2]/div[19]')
        interst.click()

        finish = self.driver.find_element_by_xpath('//*[@id="required-fields-modal"]/div[2]/button[2]')
        finish.click()

        sleep(0.5)


    def getDownloadButtons(self):
        return self.driver.find_elements_by_xpath('//*[@id="download-controls"]/div')

    def downloadModel(self, index):
        xpath = '//*[@id="app"]/div[1]/main/div/div[4]/div[2]/div[{}]/div/div/div[2]/div/div[2]/div[1]'.format(index + 1)
        download_btn = self.driver.find_element_by_xpath(xpath)

        #scroll to position
        actions = ActionChains(self.driver)
        actions.move_to_element(download_btn).perform()
        sleep(0.2)

        download_btn.click()
        try:
            download_collada_btn = self.driver.find_element_by_xpath('//*[@id="download-option-zip"]')
            download_collada_btn.click()
            return True
        except Exception:
            return False

    def getImageUrl(self, index):
        xpath = '//*[@id="app"]/div[1]/main/div/div[4]/div[2]/div[{}]/div/div/div[1]/div/picture/source[1]'.format(index + 1)
        image = self.driver.find_element_by_xpath(xpath)
        return image.get_attribute('srcset')

    def downloadImage(self, url, model_id):
        os.makedirs(os.path.join(WORKING_DIR,self.category, model_id), exist_ok=True)
        save_path = os.path.join(WORKING_DIR,self.category, model_id, 'thumbnail.webp')
        myfile = requests.get(url)
        with open(save_path, 'wb') as f:
            f.write(myfile.content)
        im = Image.open(save_path).convert("RGB")
        im.save(save_path.replace("thumbnail.webp", "thumbnail.png"), "png")
        os.remove(save_path)

    def get_model_id_from_image(self, image_url):
        return os.path.split(image_url)[1]

    def process_downloaded_element(self, model_id):
        zips = Utils.listFiles(os.path.join(DOWNLOADS_DIR,"tmp"), ".zip")
        if len(zips) > 0:
            self.extractZip(zips[0], model_id)
            os.remove(zips[0])

    def extractZip(self, path, model_id):
        print("extract {}".format(path))
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(WORKING_DIR,self.category, model_id))

    def runDownloads(self, category):
        self.category = category
        self.delete_tmp_folder()
        self.create_dirs()
        self.load_category()

        log_file = os.path.join(WORKING_DIR, self.category, 'models.txt')

        self.scroll_to_bottom()

        views = self.getDownloadButtons()
        for index, view in enumerate(views):
            try:
                self.few_questions_dialog()
            except Exception:
                pass

            #download model first
            image_url = self.getImageUrl(index)
            model_id = self.get_model_id_from_image(image_url)

            print("{} / {} : {}".format(index, len(views), model_id))

            # download files if not previously already downloaded
            with open(log_file) as f:
                if model_id in f.read():
                    print("skip {}".format(model_id))
                    continue

            successful = self.downloadModel(index)
            if successful:
                sleep(0.5)
                self.process_downloaded_element(model_id)
                self.downloadImage(image_url, model_id)
                with open(log_file, "a") as f:
                    f.write(model_id + "\n")

class Utils(object):
    def listFiles(dir, ext, ignoreExt=None):
        """
        Return array of all files in dir ending in ext but not ignoreExt.
        """
        matches = []
        for root, dirs, files in os.walk(dir):
            for f in files:
                if f.endswith(ext):
                    if not ignoreExt or (ignoreExt and not f.endswith(ignoreExt)):
                        matches.append(os.path.join(root, f))
        return matches
