# VQA
# A Baseline for Toloka Visual Question Answering Challenge at WSDM Cup 2023

**Task**: Given an image and a textual question, draw the bounding box around the object correctly responding to that question.

| Question | Image and Answer |
| --- | --- |
| What do you use to hit the ball? | <img src="https://tlk-infra-front.azureedge.net/portal-static/images/wsdm2023/tennis/x2/image.webp" width="228" alt="What do you use to hit the ball?"> |
| What do people use for cutting? | <img src="https://tlk-infra-front.azureedge.net/portal-static/images/wsdm2023/scissors/x2/image.webp" width="228" alt="What do people use for cutting?"> |
| What do we use to support the immune system and get vitamin C? | <img src="https://tlk-infra-front.azureedge.net/portal-static/images/wsdm2023/juice/x2/image.webp" width="228" alt="What do we use to support the immune system and get vitamin C?"> |

- **Competition:** <https://toloka.ai/challenges/wsdm2023>
- **CodaLab:** <https://codalab.lisn.upsaclay.fr/competitions/7434>
- **Dataset:** <https://doi.org/10.5281/zenodo.7057740>

## Configuration

```
install Docker. See the <a href="https://docs.docker.com/engine/install/">link</a>

docker pull haoyuzhang6/wsdm2023:latest

mkdir output
docker run --rm -it --gpus all --network host -v /ABSOLUTE_PATH_TO/WSDMCup2023/reproduction/data:/mnt/data -v /ABSOLUTE_PATH_TO/reproduction/output:/mnt/output wsdm2023
```
The input file will be stored in `/mnt/data/test.csv` and the input images will be at `/mnt/data/imgs`. The solution must write a single file to `/mnt/output/answer.csv`  .
