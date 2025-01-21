from dotenv import load_dotenv
import os

#testing env variables
load_dotenv('C:/Users/PJRei/rowing-analytics/.env')
print("Password:", os.getenv('PASSWORD'))
