import json
from typing import Literal

"""
Use Claude to generate a dataset of 1000 high quality samples.
"""

instruction = """
You are an expert crypto financial analyst with deep knowledge of market dynamics and sentiment analysis.
Analyze the following news story and determine its potential impact on crypto asset prices.
Focus on both direct mentions and indirect implications for each asset.

Do not output data for a given coin if the news is not relevant to it.

## Example input news story
"Goldman Sachs wants to invest in Bitcoin and Ethereum, but not in XRP"

## Example output
[
    {"coin": "BTC", "signal": 1},
    {"coin": "ETH", "signal": 1},
    {"coin": "XRP", "signal": -1},
]
"""


def generate_dataset(
    model_provider: Literal['anthropic', 'ollama'],
    n: int,
    output_file: str,
) -> list[dict]:
    # load dataset
    import pandas as pd

    df = pd.read_csv('data/cryptopanic_news.csv')
    news = df['title'].tolist()

    # random sample n news
    import random

    news = random.sample(news, n)

    from llms.factory import get_llm

    llm = get_llm(model_provider=model_provider)

    from tqdm import tqdm

    for news_item in tqdm(news):
        try:
            signals = llm.get_signal(news_item)

            output = {
                'instruction': instruction,
                'input': news_item,
                'output': signals.model_dump_json(),
                'teacher_model_name': llm.model_name,
            }

            # breakpoint()

            # append to file
            with open(output_file, 'a') as f:
                f.write(json.dumps(output) + '\n')
        except Exception as e:
            print(f'Error processing news item: {e}')
            continue


if __name__ == '__main__':
    generate_dataset(
        model_provider='anthropic',
        n=1000,
        output_file='./data/golden_dataset_anthropic.jsonl',
    )
