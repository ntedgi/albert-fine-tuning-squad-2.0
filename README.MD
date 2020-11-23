# the Albert Model for question answering fine tuning squad 2.0

## How I Run This Project Locally ?

- clone this repository

- Just Want to Play ?
    - Run : `bash install.sh`
    - Run : `bash run.sh`

 - request: ` curl -X GET \                                                                                                                
  http://localhost:8000/predict \
  -H 'Content-Type: text/plain' \
  -d '{"question":"Who was Jim Henson?", "paragraphs":[ {"id": 1, "text": "has a nice car. Jim Henson was a nice puppet."} , {"id":2,"text": "All the 2023(GoF) design patterns implemented inJavascript.Jim Henson was a monkey king"}]}'
 `
 
 - result: `{
  "result": [
    [
      1, 
      "a nice puppet"
    ], 
    [
      2, 
      "a monkey king"
    ]
  ]
}`
