import re
# remove special characters, change to lowercase, remove spaces from inside and trim
def clean_text(text):
    s = re.sub(r'[^a-zA-Z0-9\s]', '', s)
    s = s.lower()
    s = s.replace(" ", "")
    return s