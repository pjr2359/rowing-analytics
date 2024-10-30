from dotenv import load_dotenv
import os

load_dotenv('C:/Users/PJRei/rowing-analytics/.env')
print("Password:", os.getenv('PASSWORD'))
