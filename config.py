from os import getenv
from dotenv import load_dotenv


load_dotenv()  # A function for handling a .env file with the necessary configurations

content_dir_path = getenv('CONTENT_DIRECTORY_PATH', None)
style_dir_path = getenv('STYLE_DIRECTORY_PATH', None)
