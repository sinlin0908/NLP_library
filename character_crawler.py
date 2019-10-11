import requests
from bs4 import BeautifulSoup


domain = "http://www.fantiz5.com/map.html"


class InvalidUrl(Exception):
    pass


def get_web(url: str):
    response = requests.get(url)
    response.encoding = 'utf-8'

    if response.status_code != requests.codes.ok:
        raise InvalidUrl(f"invalid url:{url}")

    return response.text


def get_and_parse_web(url):
    web = get_web(url)

    soup = BeautifulSoup(web, 'lxml')

    return soup


def get_all_characters(url):

    web_content = get_and_parse_web(url)
    pages_tags = web_content.find('div', class_='daquanfenye').find_all("a")
    pages = [p.text.strip() for p in pages_tags]

    characters = []

    characters += get_one_page_characters(web_content)

    for page in pages:
        next_url = f"http://www.fantiz5.com/map_{page}.html"
        web_content = get_and_parse_web(next_url)
        characters += get_one_page_characters(web_content)

    return characters


def get_one_page_characters(web_content):

    character_table = web_content.find(
        'div', class_='daquanlist').find_all('li')

    characters = [c.text for c in character_table]

    return characters


if __name__ == "__main__":
    characters = get_all_characters(domain)

    with open("./embedding/chinese_character.text", 'w') as f:
        f.write(str(len(characters)))
        for c in characters:
            f.write(f"\n{c}")
