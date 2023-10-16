# utiliser la commande 'docker search python' pour connaitre l'image de base Python à utiliser
FROM python:3.11.4

COPY . .

# utiliser pip ou pip3, selon la version que vous avez installé
RUN pip3 install -r requirements.txt 

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"]