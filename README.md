![alt text](wikitomongo.jpg)

# Import Wikipedia Text into Mongodb
In this repository, we try to import all wikipedia articles in a database with a very easy to use acces class.
It is a perfect API for all the applications that need a fast access wikipedia database.

### 1. Download Wiki dump
Download the latest wikipedia dump from here:
https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

### 2. Install requirements
Install requirements:
```
pip install -r requirements.txt
```

### 3. Convert XML to json

Convert XML dump to json. They will be saved in clean_wiki_text_json/ path:
```
python WikiExtractor.py -o clean_wiki_text_json --json --no_templates --processes 12 enwiki-latest-pages-articles.xml.bz2
```

### 4. Fill database
Before running the command below, make sure you have your [mongodb](https://www.mongodb.com/) installed and running:

```
mongod --dbpath=path_to_database
```

Run the following command to fill the database:
```
python main.py fill_database [--path_to_config_file]
```
The --path_to_config_file is optional and you can remove it as long as you have the config file in the same directory as your main.py
Note: 
- This will take a lot of space on the disk. Make sure to have enough free space before running.
- The progress bar shows number of files not articles
- This takes about .. hours to finish.

### 4. Create index on articles (Optional) **Strongly recommened**
Create index on articles

```
python main.py create_index
```
This is very important for faster access of articles in the database.

### Usage
The [WikiDatabase](https://github.com/roholazandie/wikipedia_to_mongodb/blob/master/wiki_database.py) class is designed to be like an iterable. You can access to it like a list:
```python
from wiki_database import WikiDatabase

config_file="wiki_database_config.json"
wiki_database = WikiDatabase(config_file)
print(wiki_database[100])
```
Or iterate over it:

```python
from wiki_database import WikiDatabase

config_file="wiki_database_config.json"
wiki_database = WikiDatabase(config_file)
for article in wiki_database:
    print(article["article"])
```

The article is a dictionary with four keys:
- "article": the id number of the article
- "title": the title of the wikipedia article
- "text": the actual text of the wikipedia article
- "pageid": the pageid provided by wikipedia for each article

Credit:
This repository is strongly based on two repositories:
 - [WikiExtractor](https://github.com/attardi/wikiextractor)
 - [Gensim](https://github.com/RaRe-Technologies/gensim)