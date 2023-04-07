import spacy
import string
nlp = spacy.load("en_core_web_sm")


def normalize_text(raw_text):
    raw_text = str(raw_text).lower()
    text = raw_text.translate(str.maketrans('','',string.punctuation))
    text = nlp(text)
    final_token_list = [token for token in text if token.is_stop == False]
    final_text_list = [token.text for token in final_token_list]
    text_processed = " ".join(final_text_list)
    
    return text_processed



aaaaa = "@VirginAmerica it's really aggressive to blast obnoxious ""entertainment"" in your guests' faces &amp; they have little recourse"
normalize_text(aaaaa)