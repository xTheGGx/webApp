import sys
sys.path.insert(0,"..")

from src.api import app

if __name__ == "__main__":
    app.run()