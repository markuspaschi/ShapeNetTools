# Downloader for 3D Warehouse

This Database is hosted by Google and contains a large Database of obj, dae and other Model files.

Check [3D Warehouse](https://3dwarehouse.sketchup.com) for more information.

Because 3D Warehouse does not have an open API, this downloader utilizes selenium for chrome to download one object after another.

Built with Python, Selenium, and Google Chrome.

## Requirements

* Chrome Browser
* download [chromedriver](https://chromedriver.chromium.org), unzip, move to /usr/local/bin (mac os / linux)
* pip install selenium

## Fill out

You need a valid 3D Warehouse Account!

> DOWNLOADS_DIR = ""  
WORKING_DIR = ""  
EMAIL = ""  
PASSWORD = ""

## Run

Interactive python is recommended, but any method works.

For example:

```
-python -i downloader.py
wh = Warehouse()
wh.runDownloads("MY_CATEGORY_FOR_DOWNLAODED_OBJECTS")
```

Adapt the code, if you want to filter more specific on the website and run the downloads afterwards.


This will log you in with your specified credentials, and download all objects into your download/working directory. Leave chrome in foreground!
