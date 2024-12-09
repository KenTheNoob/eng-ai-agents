# See README for more info on how the DataCollectionPipeline works
# The ETL pipeline is part of the DataCollectionPipeline
# Remove the time.sleep(1) line if you are sure you won't get blocked from a webpage for requesting too often
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse

import pymongo
import requests
from bs4 import BeautifulSoup
from clearml import PipelineDecorator
from dotenv import load_dotenv

# Setup ClearML
try:
    load_dotenv(override=True)
except Exception:
    load_dotenv(sys.path[1] + "/.env", override=True)
CLEARML_WEB_HOST = os.getenv("CLEARML_WEB_HOST")
CLEARML_API_HOST = os.getenv("CLEARML_API_HOST")
CLEARML_FILES_HOST = os.getenv("CLEARML_FILES_HOST")
CLEARML_API_ACCESS_KEY = os.getenv("CLEARML_API_ACCESS_KEY")
CLEARML_API_SECRET_KEY = os.getenv("CLEARML_API_SECRETKEY")

# Input into the Data Collection Pipeline is a list of links to domains
"""
links = [
    "https://www.ros.org/",
    "https://docs.nav2.org/",
    "https://moveit.ai/",
    "https://gazebosim.org/home",
    "https://github.com/ros2/ros2",
    "https://github.com/ros-navigation/navigation2",
    "https://github.com/moveit/moveit2",
    "https://github.com/gazebosim/gazebo-classic",
]
"""
links = ["https://www.ros.org/", "https://github.com/ros2/ros2"]


# ETL pipeline
@PipelineDecorator.component(cache=False, return_values=["documents, codes"])
def ETL_Pipeline(links):
    # Create a mongoDB connection to check for duplicates before inserting
    try:
        load_dotenv(override=True)
    except Exception:
        load_dotenv(sys.path[1] + "/.env", override=True)
    DATABASE_HOST = os.getenv("DATABASE_HOST")
    mongoHost = pymongo.MongoClient(DATABASE_HOST)
    mongoDatabase = mongoHost["twin"]

    # Extract data from links and their subdirectories(using crawlers)
    documents = []
    codes = []
    for link in links:
        # Web scraper/crawler for github links
        if "https://github.com" in link:
            # Do not revisit a link already in the database
            mongoCollection = mongoDatabase["Github"]
            result = mongoCollection.find_one({"link": link})
            if result is None:
                # Modified GithubCrawler from LLM-Engineer for scraping github
                local_temp = tempfile.mkdtemp()
                try:
                    os.chdir(local_temp)
                    subprocess.run(["git", "clone", link])
                    repo_path = os.path.join(local_temp, os.listdir(local_temp)[0])
                    tree = {}
                    for root, _, files in os.walk(repo_path):
                        dir = root.replace(repo_path, "").lstrip("/")
                        if dir.startswith((".git", ".toml", ".lock", ".png")):
                            continue
                        for file in files:
                            if file.endswith((".git", ".toml", ".lock", ".png")):
                                continue
                            file_path = os.path.join(dir, file)
                            with open(
                                os.path.join(root, file), "r", errors="ignore"
                            ) as f:
                                tree[file_path] = f.read().replace(" ", "")
                except Exception:
                    print(f"Error scrapping {link}")
                finally:
                    shutil.rmtree(local_temp)
                    # Correct the link
                    r = requests.get(link)
                    soup = BeautifulSoup(r.content, "html.parser")
                    # Find the file path to any of the files in the repository
                    link_element = soup.find("a", attrs={"class": "Link--primary"})
                    path = link_element.get("href")
                    path = path.rsplit("/", 1)[0]
                    # Push all the subdirectories to mongo
                    for subdirectory in tree:
                        text = tree[subdirectory]
                        # Transform the data
                        # Get rid of repeating \n characters and spaces
                        text = text.replace("\t", " ")
                        text = text.replace("\n", " ")
                        text_len = len(text)
                        for i in range(text_len):
                            while (
                                i + 1 < text_len
                                and text[i] == " "
                                and text[i + 1] == " "
                            ):
                                text = text[:i] + text[i + 1 :]
                                text_len -= 1
                        codes.append(
                            {
                                "link": "https://github.com"
                                + path
                                + "/"
                                + subdirectory,
                                "type": "Github",
                                "content": text,
                            }
                        )
        # Web scraper/crawler for other links(Documents)
        else:
            # Do not revisit a link already in the database
            mongoCollection = mongoDatabase["Document"]
            result = mongoCollection.find_one({"link": link})
            if result is None:
                try:
                    # Get all text in the website
                    r = requests.get(link)
                    soup = BeautifulSoup(r.content, "html.parser")
                    soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
                    text = soup.get_text()
                    # Transform the data
                    # Get rid of repeating \n characters and spaces
                    text = text.replace("\t", " ")
                    text = text.replace("\n", " ")
                    text_len = len(text)
                    for i in range(text_len):
                        while (
                            i + 1 < text_len and text[i] == " " and text[i + 1] == " "
                        ):
                            text = text[:i] + text[i + 1 :]
                            text_len -= 1
                    if "404" not in text:
                        documents.append(
                            {"link": link, "type": "Document", "content": text}
                        )
                    # Also crawl through all subdirectorys in the link(related links)
                    soup = BeautifulSoup(r.content, "html.parser")
                    subdirectories = [a.get("href") for a in soup.find_all("a")]
                    for subdirectory in subdirectories:
                        newLink = urllib.parse.urljoin(link, subdirectory)
                        if (
                            subdirectory is not None
                            and "http" not in subdirectory
                            and "#" not in subdirectory
                            and ".zip" not in subdirectory
                            and ".pdf" not in subdirectory
                            and mongoCollection.find_one({"link": newLink}) is None
                            and newLink not in links
                        ):
                            links.append(newLink)
                except Exception:
                    print("Could not crawl link", link)
        # Avoid spamming sites
        time.sleep(0.1)
    # Each document has a link, type(github or other), and content(text)
    mongoCollection = mongoDatabase["Document"]
    mongoCollection.insert_many(documents)
    mongoCollection = mongoDatabase["Github"]
    mongoCollection.insert_many(codes)
    return documents, codes


# Allow ClearML to monitor and run the ETL pipeline
@PipelineDecorator.pipeline(
    name="Data Collection Pipeline",
    project="RAG LLM",
    version="0.4",
)
def main():
    return ETL_Pipeline(links)


if __name__ == "__main__":
    PipelineDecorator.run_locally()
    main()
