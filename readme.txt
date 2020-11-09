The default port is 4000
To deploy go to the project directory run :

docker build -t nlp .

docker run -p 4000:4000 -d --name mynlp nlp 

