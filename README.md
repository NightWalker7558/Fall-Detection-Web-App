# Fall Detection

This project uses pre-trained YOLOv8 models that have been fine-tuned with various datasets from Roboflow for fall detection.

The project consists of a client and a server. Follow the instructions below to set up and run the project.

## Client

The client is a React application. To run the client, navigate to the client directory and install the necessary dependencies:

```bash
cd client
npm install
```

After the dependencies are installed, you can start the client:

```bash
npm run start
```

## Server

The server is a Flask application. To run the server, navigate to the server directory and install the necessary dependencies:

```bash
cd server
pip install -r requirements.txt
```

After the dependencies are installed, you can start the server:

```bash
flask run
```