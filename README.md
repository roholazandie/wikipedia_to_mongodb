
### 1. Download Wiki dump
Download the latest wikipedia dump from here:
https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

### 2. Convert XML to json

Convert XML dump to json. They will be saved in clean_wiki_text_json/ path:
```
python WikiExtractor.py -o clean_wiki_text_json --json --no_templates --processes 12 enwiki-latest-pages-articles.xml.bz2
```

### 3. Fill database
Run the following command to fill the database:
```
python main.py fill_database [--path_to_config_file]
```
The --path_to_config_file is optional and you can remove it as long as you have the config file in the same directory as your main.py
Note: This will take a lot of space on the disk. Make sure to have enough free space before running.

### 4. Create index on articles (Optional) **Strongly recommened**
Create index on articles

```
python main.py create_index
```
This is very important for faster access of articles in the database.

### Usage
The WikiDatabase class is designed to be like an iterable. You can access to it like a list:
```
config_file="wiki_database_config.json"
wiki_database = WikiDatabase(config_file)
wiki_database[100]
```
Or iterate over it:

```
config_file="wiki_database_config.json"
wiki_database = WikiDatabase(config_file)
for article in wiki_database:
    print(article["article"]
```

The article is a dictionary with four keys:
- "article": the id number of the article
- "title": the title of the wikipedia article
- "text": the actual text of the wikipedia article
- "pageid": the pageid provided by wikipedia for each article

Credit:
This repository is strongly based on two repositories:
 - https://github.com/attardi/wikiextractor
 - https://github.com/RaRe-Technologies/gensim