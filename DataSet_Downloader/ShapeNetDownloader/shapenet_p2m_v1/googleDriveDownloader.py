import os
import requests
from bs4 import BeautifulSoup 

class googleDriveFileDownloader():
    
    def __init__(self):
        Flag = True

    def downloadFile(self,url):
        if(url.startswith("https://drive.google.com/uc")):
            print("Download is starting")
            URL = url
            r = requests.get(URL) 

            Fileid = URL.split("=")[1].split("&")[0]

            soup = BeautifulSoup(r.content, 'html.parser') 

            FileName = soup.select('.uc-name-size')

            print(FileName[0].select('a')[0].text)

            url = r'wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id='+str(Fileid)+'" -O- | sed -rn "s/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p")&id='+str(Fileid)+'" -O '+str(FileName[0].select('a')[0].text)+' && rm -rf /tmp/cookies.txt'

            return(os.system(url))
        else:
            print("Unable to process the URL \nMake sure to have it in this format 'https://drive.google.com/uc?id=****&export=download'")
            return False
