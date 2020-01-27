from wiki_database import WikiDatabase
import fire


def fill_database(config_file="wiki_database_config.json"):
    wiki_database = WikiDatabase(config_file)
    wiki_database.populate_database()



def create_index(config_file="wiki_database_config.json"):
    wiki_database = WikiDatabase(config_file)
    wiki_database.create_index("article")


def database_length(config_file="wiki_database_config.json"):
    wiki_database = WikiDatabase(config_file)
    return len(wiki_database)


def get_wikipedia_article(i, config_file="wiki_database_config.json"):
    wiki_database = WikiDatabase(config_file)
    return str(wiki_database[int(i)])

#wiki_database.populate_database()
# wiki_database.create_index("article")

# for article in wiki_database:
#     print(article["title"])

# import time
#
# t1 = time.time()
# print(wiki_database[4000])
# t2 = time.time()
# print(wiki_database[1000001])
# t3 = time.time()
# print(wiki_database[2000000])
# t4 = time.time()
# print(wiki_database[4000000])
# t5 = time.time()
# print(t2 - t1)
# print(t3 - t2)
# print(t4 - t3)
# print(t5 - t4)

if __name__ == "__main__":
    fire.Fire()
    # config_file = "wiki_database_config.json"
    # wiki_database = WikiDatabase(config_file)
    # print(wiki_database[100])