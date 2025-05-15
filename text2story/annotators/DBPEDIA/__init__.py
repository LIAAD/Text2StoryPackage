from bs4 import BeautifulSoup
import sys
import requests
import socket

from text2story.core.exceptions import InvalidLanguage, NoConnection


# Language mapping for DBpedia Spotlight API URL
language_to_url = {
    'en': 'http://api.dbpedia-spotlight.org/en/annotate',
    'pt': 'http://api.dbpedia-spotlight.org/pt/annotate'
}

def load(lang, key=None):
    """
    Used, at start, to load the pipeline for the supported languages.
    """
    # Checks if the language is supported
    if lang not in language_to_url:
        raise InvalidLanguage(lang)

def check_internet_connection(host="8.8.8.8", port=53, timeout=3):
    """
    Check internet connection by trying to connect to a specified host and port.
    """
    try:
        # Create a socket object
        socket.setdefaulttimeout(timeout)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Try to connect to the host and port
        s.connect((host, port))
        s.close()
        return True
    except OSError:
        return False


def extract_participants(text, lang=None, url=None):

    # Request parameters
    params = {
        "text": text,
        "confidence": 0.2,  # Minimum confidence level for annotations
        "support": 20       # Minimum number of supporting documents
    }

    if lang:
        # Check internet connection
        if not check_internet_connection():
            raise NoConnection

        # Checks if the language is supported
        if lang not in language_to_url:
            raise InvalidLanguage(lang)

        # DBpedia Spotlight API URL based on language
        url = language_to_url[lang]

    elif url:
        # DBpedia Spotlight service URL in Docker container
        url = url

        # Request parameters
        params = {
            "text": text,
            "confidence": 0.2,  # Minimum confidence level for annotations
            "support": 20       # Minimum number of supporting documents
        }

        # Making the POST request
        response = requests.post(url, data=params)
    else:
        raise BaseException(f"Error calling extract_participants. Check if you wrote all the necessary parameters in the function.")
        
    # Making the GET request
    response = requests.get(url, params=params)

    # Checks whether the request was successful
    if response.status_code == 200:
        # Parse the HTML returned by the response
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Finds all <a> tags that contain information about resources
        resource_tags = soup.find_all('a')

        # Extract information from resources
        participants = []
        for tag in resource_tags:
            actor = tag.get_text()
            start_character_offset = params['text'].find(str(actor))
            end_character_offset = start_character_offset + len(str(actor))
            actor_head = None
            actor_type = 'Other'
            participants.append(((start_character_offset, end_character_offset), actor_head, actor_type))

        return participants
    else:
        raise BaseException(f"Error calling DBpedia URL {url}. Status code: {response.status_code}")

