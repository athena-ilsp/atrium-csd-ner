# Atrium: Named Entity Recognition for Descriptions in Archaeological Context Sheets 

The application is built with FastAPI and is ready for containerization with Docker. The application leverages a NER model available at https://huggingface.co/pprokopidis/atrium-speech-based-ner/ and the Flair NLP framework https://github.com/flairNLP/flair. 

The application receives text that typically appears in context sheets as its input and executes the following steps:

- Text Segmentation: The input text is parsed and segmented into sentences and tokens to prepare it for analysis.
- Entity Extraction: Named Entity Recognition (NER) models trained on domain-specific NER are applied to the text. These models identify and classify key terms within the descriptions.
- Export of results in JSON format

## Prerequisites

- [Install Docker](https://docs.docker.com/get-docker/)
- [Install Docker Compose](https://docs.docker.com/compose/install/)


## Installation
1. Clone the repository.
```
git clone https://github.com/athena-ilsp/atrium-csd-ner.git
```
2. Navigate into the project directory and make a copy of the example environment variables file.
```
cd atrium-csd-ner
cp .env.example .env
```
The .env file should contain the HuggingFace HUGGING_FACE_HUB_TOKEN. You should add your own to this file.

## Quickstart
These steps show how to set up the application using Docker.

1. In the project root directory, build the Docker images.
```
docker-compose build
```

2. Run the Docker containers.
```
docker-compose up 
```
At this point, the application should be running at [http://localhost:8080/](http://localhost:8080/). To stop the application, you can run:
```
docker-compose down
```
3. To restart or rebuild the application, you can run:
```
docker-compose up --build
```

## Documentation
FastAPI autogenerates an OpenAPI specification, which allows you to test this application directly from an interactive console in your browser. It uses the [Pydantic](https://docs.pydantic.dev/) model to validate user input (as shown in the models section of the specification, below). Go to [http://0.0.0.0:8080/docs](http://0.0.0.0:8080/docs) to use the automatic interactive API documentation for this application (provided by [Swagger UI](https://github.com/swagger-api/swagger-ui)) to send requests. 

## Example call of the API

```bash
curl -X 'POST' \
  'http://10.1.1.76:8080/ner' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "Lower fill of posthole transitioning into fill 1036 gradually.\nAn urn found in context 526.\nProbably Late Mesolithic.\nAnimal bones in context 631.\n"
}'
```

## Example response

```json
{
  "sentences": [
    {
      "text": "Lower fill of posthole transitioning into fill 1036 gradually.",
      "ents": [
        {
          "start": 0,
          "end": 22,
          "label": "CONTEXT"
        },
        {
          "start": 42,
          "end": 51,
          "label": "CONTEXT"
        }
      ],
      "title": null
    },
    {
      "text": "An urn found in context 526.",
      "ents": [
        {
          "start": 3,
          "end": 6,
          "label": "ARTIFACT"
        },
        {
          "start": 16,
          "end": 27,
          "label": "CONTEXT"
        }
      ],
      "title": null
    },
    {
      "text": "Probably Late Mesolithic.",
      "ents": [
        {
          "start": 9,
          "end": 24,
          "label": "PERIOD"
        }
      ],
      "title": null
    },
    {
      "text": "Animal bones in context 631.",
      "ents": [
        {
          "start": 0,
          "end": 12,
          "label": "ARTIFACT"
        },
        {
          "start": 16,
          "end": 27,
          "label": "CONTEXT"
        }
      ],
      "title": null
    },
    {
      "text": "",
      "ents": [],
      "title": null
    }
  ]
}
```


## License

This project is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.
